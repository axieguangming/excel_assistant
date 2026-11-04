"""构造歧义澄清训练样本。

流程：
1. `dataset_builder.ambiguity.detect` 在每张表上找出**可验证**的歧义场景
   （两个候选在数据上确实给出不同结果）。
2. 用 LLM 为每个场景写一条自然的用户提问——问法必须使用模糊说法、
   **不得自行消解歧义**（不能出现"去重""平均值"等澄清词）。
3. 澄清措辞由程序按模板拼接，注入真实列名，保证一定点出两个具体选项。

第 2 步的产物会缓存，重跑不重复付费。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage

from dataset_builder.ambiguity import Ambiguity, detect
from dataset_builder.llm import DatasetLLM, _content_text, extract_json
from dataset_builder.tables import load_tables

# 每种歧义里，如果问法出现这些词就说明模型自己消解了歧义，需要丢弃。
DISAMBIGUATED_MARKERS: dict[str, tuple[str, ...]] = {
    "count_semantics": ("去重", "不重复", "不同", "多少种", "唯一"),
    "metric_column": (),
    "aggregation": ("平均", "均值", "合计", "总和", "求和", "加总"),
    "group_dimension": (),
    "time_range": ("最近一个月", "最近7天", "最近30天", "上月", "本月"),
    "missing_value": ("忽略空", "跳过空", "按0", "当作0", "视为0"),
    "top_n": ("前10", "前5", "前3", "前20", "前若干", "前"),
    "chart_type": ("柱状图", "饼图", "折线图", "散点图"),
}

SYSTEM = (
    "你是中文 Excel 用户的代言人。给定工作表信息和一个**待澄清的歧义点**，"
    "请写真实用户会说的提问。\n"
    "\n"
    "硬性要求：\n"
    "1. 每一条提问都必须**围绕给定的歧义点**展开，且使用给定的模糊说法"
    "（例如「金额」「最近」「画个图」）。\n"
    "2. **不能自行消解这个歧义**——不要补充具体列名、统计方式、时间窗口、"
    "图表类型或具体条数。\n"
    "3. 提问要像真人说话：可带业务目的（如「跟领导汇报」）、可省略表名，"
    "长度 15–50 字。\n"
    "4. 不要出现工具名、英文参数名或 JSON。\n"
    "5. 每条风格不同：简洁直白、口语省略、业务口吻、略啰嗦。\n"
    "\n"
    "只输出 JSON：{\"questions\": [\"...\", \"...\"]}"
)


def build_catalog(table) -> dict:
    return {
        "sheet": table.sheet,
        "columns": [str(c) for c in table.frame.columns],
        "rows": int(len(table.frame)),
    }


def describe(ambiguity: Ambiguity) -> dict:
    return {
        "kind": ambiguity.kind,
        "vague_term": ambiguity.vague_term,
        "ambiguity_explained": ambiguity.hint,
        "do_not_specify": list(ambiguity.options),
    }


def ask_questions(
    llm: DatasetLLM,
    table,
    ambiguity: Ambiguity,
    *,
    count: int,
) -> list[str]:
    """针对单个歧义生成问法。分开调用可保证每条都聚焦在该歧义。"""
    payload = {
        "worksheet": build_catalog(table),
        "ambiguity": describe(ambiguity),
        "questions_wanted": count,
    }
    response = llm.model.invoke(
        [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
    )
    text = _content_text(getattr(response, "content", ""))
    parsed = extract_json(text)
    if isinstance(parsed, dict):
        items = parsed.get("questions") or parsed.get("items") or []
    else:
        items = parsed
    if not isinstance(items, list):
        raise ValueError("提问生成结果必须是数组")
    return [str(item).strip() for item in items if str(item).strip()]


def is_usable(ambiguity: Ambiguity, question: str) -> tuple[bool, str]:
    if len(question) < 10:
        return False, "过短"
    if len(question) > 90:
        return False, "过长"
    for marker in ("aggregate_data", "filter_rows", "group_by", "chart_y", "JSON"):
        if marker in question:
            return False, "含工具名"
    if not ambiguity.matches(question):
        return False, f"未聚焦到歧义（缺 {ambiguity.vague_term}）"
    for marker in DISAMBIGUATED_MARKERS.get(ambiguity.kind, ()):
        if marker in question:
            return False, f"自行消解歧义：{marker}"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="构造歧义澄清训练样本")
    parser.add_argument("--data", default="datasets/2026-10-29")
    parser.add_argument("--per-kind", type=int, default=1, help="每个歧义生成几条问法")
    parser.add_argument("--cache", default="build/ambiguity-questions.json")
    parser.add_argument("--out", default="build/clarifications.json")
    parser.add_argument("--limit-tables", type=int, default=0)
    args = parser.parse_args()

    tables = load_tables(Path(args.data) / "tables")
    if args.limit_tables:
        tables = tables[: args.limit_tables]

    detected: dict[str, list[Ambiguity]] = {}
    for table in tables:
        found = detect(table)
        if found:
            detected[table.table_id] = found
    total_amb = sum(len(v) for v in detected.values())
    print(f"表格 {len(tables)} 张，检出歧义 {total_amb} 个")

    cache_path = Path(args.cache)
    cache: dict[str, dict] = {}
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"复用问法缓存：{len(cache)} 个键")

    llm = DatasetLLM()
    # 缓存键 = 工作簿 + 歧义类型；逐歧义调用可保证每条问法都聚焦。
    jobs: list[tuple[str, Ambiguity]] = [
        (table_id, ambiguity)
        for table_id, ambiguities in detected.items()
        for ambiguity in ambiguities
    ]
    pending = [
        (table_id, ambiguity)
        for table_id, ambiguity in jobs
        if f"{table_id}||{ambiguity.kind}" not in cache
    ]
    print(f"需要调用 LLM 的歧义数：{len(pending)} / {len(jobs)}")

    started = time.time()
    table_by_id = {table.table_id: table for table in tables}
    for index, (table_id, ambiguity) in enumerate(pending, start=1):
        try:
            questions = ask_questions(
                llm,
                table_by_id[table_id],
                ambiguity,
                count=args.per_kind,
            )
        except Exception as error:  # noqa: BLE001 - 单个失败不中断
            print(f"  [{index}/{len(pending)}] {table_id}/{ambiguity.kind} 失败：{error}")
            continue
        cache[f"{table_id}||{ambiguity.kind}"] = questions
        if index % 20 == 0 or index == len(pending):
            print(
                f"  {index}/{len(pending)}  已用 {time.time() - started:.0f}s",
                flush=True,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False), encoding="utf-8"
            )

    # ---- 组装样本 ----
    samples: list[dict] = []
    rejected = Counter()
    kind_counts = Counter()
    for table in tables:
        for ambiguity in detected.get(table.table_id) or []:
            questions = cache.get(f"{table.table_id}||{ambiguity.kind}") or []
            for offset, question in enumerate(questions):
                ok, reason = is_usable(ambiguity, question)
                if not ok:
                    rejected[reason] += 1
                    continue
                samples.append(
                    {
                        "workbook_id": table.table_id,
                        "kind": ambiguity.kind,
                        "user_request": question,
                        "expected_response": ambiguity.render(offset),
                        "expected_actions": [],
                        "task_categories": ["clarify"],
                        "evidence": ambiguity.evidence,
                    }
                )
                kind_counts[ambiguity.kind] += 1

    out = Path(args.out)
    out.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n=== 生成结果 ===")
    print(f"  耗时            {time.time() - started:.0f}s")
    print(f"  澄清样本        {len(samples)}")
    print(f"  被过滤          {sum(rejected.values())}")
    for reason, count in rejected.most_common():
        print(f"    {reason:28s} {count}")
    print(f"  覆盖工作簿      {len({s['workbook_id'] for s in samples})}")
    print(f"  类型分布        {dict(kind_counts)}")
    print(f"  输出            {out}")

    print("\n=== 样例 ===")
    shown: Counter[str] = Counter()
    for sample in samples:
        if shown[sample["kind"]] >= 1:
            continue
        shown[sample["kind"]] += 1
        print(f"\n  [{sample['kind']}] {sample['workbook_id']}")
        print(f"    问: {sample['user_request']}")
        print(f"    澄清: {sample['expected_response']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
