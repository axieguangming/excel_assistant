"""仿真表格生成：覆盖常见业务场景与可控的数据缺陷。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path
import random
from typing import Any, Iterable

import pandas as pd

DOMAINS: tuple[str, ...] = (
    "sales_orders",
    "inventory",
    "staffing",
    "budget",
    "survey",
)

SHEET_NAMES: dict[str, str] = {
    "sales_orders": "订单明细",
    "inventory": "库存",
    "staffing": "排班",
    "budget": "预算执行",
    "survey": "问卷",
}

SOURCE: dict[str, str] = {"type": "synthetic", "license_id": "synthetic-internal"}


def column_kind(series: pd.Series) -> str:
    """把列归为 pipeline 使用的四类：numeric/datetime/boolean/text。"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "text"


@dataclass(frozen=True)
class SyntheticTable:
    table_id: str
    domain: str
    sheet: str
    frame: pd.DataFrame

    def profile(self) -> dict[str, Any]:
        """生成给大模型使用的工作表概况；仿真数据不含真实个人信息。"""
        columns: list[dict[str, Any]] = []
        for name in self.frame.columns:
            series = self.frame[name]
            present = series.dropna()
            columns.append(
                {
                    "name": str(name),
                    "kind": column_kind(series),
                    "non_null": int(present.size),
                    "unique": int(present.nunique()),
                    "samples": [str(value)[:24] for value in present.head(3)],
                }
            )
        return {
            "table_id": self.table_id,
            "domain": self.domain,
            "sheet": self.sheet,
            "rows": int(len(self.frame)),
            "columns": columns,
            "missing_cells": int(self.frame.isna().sum().sum()),
        }


def _nullify(frame: pd.DataFrame, column: str, rng: random.Random, rate: float = 0.06) -> None:
    series = frame[column]
    if pd.api.types.is_integer_dtype(series):
        series = series.astype("Int64")
    elif pd.api.types.is_bool_dtype(series):
        series = series.astype("boolean")
    mask = pd.Series(
        [rng.random() < rate for _ in range(len(frame))],
        index=frame.index,
    )
    frame[column] = series.mask(mask, other=pd.NA)


def _finalize(
    rng: random.Random,
    frame: pd.DataFrame,
    *,
    nullable: Iterable[str] = (),
    duplicate_rows: bool = True,
    all_empty_column: bool = True,
) -> pd.DataFrame:
    for column in nullable:
        if column in frame.columns:
            _nullify(frame, column, rng)

    if duplicate_rows and len(frame) > 6 and rng.random() < 0.45:
        copied = min(2, max(1, len(frame) // 20))
        extra = frame.iloc[[rng.randrange(len(frame)) for _ in range(copied)]]
        frame = pd.concat([frame, extra], ignore_index=True)

    if all_empty_column and rng.random() < 0.4:
        frame["备注"] = pd.Series([pd.NA] * len(frame), dtype="object")

    return frame.reset_index(drop=True)


def _sales_orders(rng: random.Random) -> pd.DataFrame:
    rows = rng.randint(24, 110)
    start = date(2025, 1, 1)
    frame = pd.DataFrame(
        {
            "订单号": [f"SO{1000 + index}" for index in range(rows)],
            "日期": [start + timedelta(days=rng.randint(0, 364)) for _ in range(rows)],
            "区域": [rng.choice(["华东", "华北", "华南", "西南"]) for _ in range(rows)],
            "产品线": [rng.choice(["A型", "B型", "C型"]) for _ in range(rows)],
            "销售员": [rng.choice(["张伟", "李娜", "王强", "赵敏"]) for _ in range(rows)],
            "数量": pd.array([rng.randint(1, 40) for _ in range(rows)], dtype="Int64"),
            "单价": [round(rng.uniform(8, 400), 2) for _ in range(rows)],
        }
    )
    frame["日期"] = pd.to_datetime(frame["日期"])
    frame["销售额"] = (frame["数量"].astype("float64") * frame["单价"]).round(2)
    frame["折扣"] = [round(rng.uniform(0, 0.2), 3) for _ in range(rows)]
    return _finalize(rng, frame, nullable=["单价", "销售额", "区域"])


def _inventory(rng: random.Random) -> pd.DataFrame:
    rows = rng.randint(20, 90)
    start = date(2025, 3, 1)
    frame = pd.DataFrame(
        {
            "仓库": [rng.choice(["北仓", "南仓", "中心仓"]) for _ in range(rows)],
            "商品编码": [f"SKU{2000 + index}" for index in range(rows)],
            "商品名称": [rng.choice(["螺丝", "垫片", "轴承", "外壳", "线束"]) for _ in range(rows)],
            "分类": [rng.choice(["标准件", "结构件", "电子件"]) for _ in range(rows)],
            "库存数量": pd.array([rng.randint(0, 800) for _ in range(rows)], dtype="Int64"),
            "安全库存": pd.array([rng.randint(10, 200) for _ in range(rows)], dtype="Int64"),
            "单价": [round(rng.uniform(0.5, 120), 2) for _ in range(rows)],
            "最近入库": [
                start + timedelta(days=rng.randint(0, 200)) for _ in range(rows)
            ],
        }
    )
    frame["最近入库"] = pd.to_datetime(frame["最近入库"])
    return _finalize(rng, frame, nullable=["库存数量", "单价", "分类"])


def _staffing(rng: random.Random) -> pd.DataFrame:
    rows = rng.randint(22, 80)
    start = date(2025, 6, 1)
    frame = pd.DataFrame(
        {
            "员工编号": [f"E{100 + index}" for index in range(rows)],
            "姓名": [f"员工{index + 1}" for index in range(rows)],
            "部门": [rng.choice(["客服", "仓储", "运营", "技术"]) for _ in range(rows)],
            "班次": [rng.choice(["早班", "中班", "晚班"]) for _ in range(rows)],
            "日期": [start + timedelta(days=rng.randint(0, 60)) for _ in range(rows)],
            "工时": [round(rng.uniform(4, 10), 1) for _ in range(rows)],
            "加班时长": [round(rng.uniform(0, 3), 1) for _ in range(rows)],
        }
    )
    frame["日期"] = pd.to_datetime(frame["日期"])
    return _finalize(rng, frame, nullable=["班次", "加班时长"])


def _budget(rng: random.Random) -> pd.DataFrame:
    rows = rng.randint(20, 70)
    frame = pd.DataFrame(
        {
            "部门": [rng.choice(["市场", "研发", "行政", "销售"]) for _ in range(rows)],
            "科目": [rng.choice(["差旅", "物料", "外包", "培训"]) for _ in range(rows)],
            "月份": [f"2025-{rng.randint(1, 12):02d}" for _ in range(rows)],
            "预算金额": [round(rng.uniform(1000, 90000), 2) for _ in range(rows)],
            "实际支出": [round(rng.uniform(500, 120000), 2) for _ in range(rows)],
        }
    )
    frame["差异"] = (frame["实际支出"] - frame["预算金额"]).round(2)
    return _finalize(rng, frame, nullable=["实际支出", "科目"])


def _survey(rng: random.Random) -> pd.DataFrame:
    rows = rng.randint(25, 100)
    feedback = ["服务及时", "等待较久", "流程清晰", "需要改进", "整体满意", ""]
    frame = pd.DataFrame(
        {
            "受访者编号": [f"R{500 + index}" for index in range(rows)],
            "年龄段": [rng.choice(["18-25", "26-35", "36-45", "46+"]) for _ in range(rows)],
            "城市": [rng.choice(["上海", "北京", "成都", "广州"]) for _ in range(rows)],
            "满意度": pd.array([rng.randint(1, 5) for _ in range(rows)], dtype="Int64"),
            "推荐意愿": pd.array([rng.randint(0, 10) for _ in range(rows)], dtype="Int64"),
            "反馈": [rng.choice(feedback) for _ in range(rows)],
        }
    )
    return _finalize(rng, frame, nullable=["满意度", "城市", "反馈"])


_GENERATORS = {
    "sales_orders": _sales_orders,
    "inventory": _inventory,
    "staffing": _staffing,
    "budget": _budget,
    "survey": _survey,
}


def generate_table(domain: str, seed: int) -> SyntheticTable:
    if domain not in _GENERATORS:
        raise ValueError(f"未知的仿真场景：{domain}。可选：{', '.join(DOMAINS)}")
    rng = random.Random(seed)
    frame = _GENERATORS[domain](rng)
    return SyntheticTable(
        table_id=f"{domain}-{seed:04d}",
        domain=domain,
        sheet=SHEET_NAMES[domain],
        frame=frame,
    )


def generate_tables(
    count: int,
    *,
    seed: int = 20251023,
    domains: Iterable[str] | None = None,
) -> list[SyntheticTable]:
    """按场景轮转生成表格，保证同一批数据覆盖多个业务结构。"""
    if count < 1:
        raise ValueError("表格数量必须大于 0。")
    chosen = list(domains) if domains else list(DOMAINS)
    unknown = [domain for domain in chosen if domain not in _GENERATORS]
    if unknown:
        raise ValueError(f"未知的仿真场景：{unknown}")

    tables: list[SyntheticTable] = []
    for index in range(count):
        domain = chosen[index % len(chosen)]
        tables.append(generate_table(domain, seed + index))
    return tables


def _excel_sheet_name(name: str) -> str:
    """Excel 工作表名不能含 []:*?/\\ 且不超过 31 个字符。"""
    cleaned = "".join("_" if char in "[]:*?/\\" else char for char in name).strip()
    return (cleaned or "工作表")[:31]


def _write_table(
    frame: pd.DataFrame,
    path: Path,
    *,
    table_format: str,
    sheet_name: str,
) -> None:
    if table_format == "xlsx":
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name=sheet_name)
    elif table_format == "csv":
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError(f"不支持的表格格式：{table_format}")


def _read_table(path: Path, table_format: str, sheet_name: str) -> pd.DataFrame:
    """按格式读回表格，与用户上传 Excel 时的读取路径保持一致。"""
    if table_format == "xlsx":
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_csv(path, encoding="utf-8-sig")


def save_tables(
    tables: Iterable[SyntheticTable],
    directory: str | Path,
    *,
    table_format: str = "xlsx",
) -> Path:
    """把表格写成 Excel（默认）或 CSV，并记录写盘后的真实列类型。

    索引中的列类型取自**写盘后重新读取**的表格，而不是内存中的生成结果，
    因为 Excel 往返会改变部分 dtype（例如可空整数变成浮点）。
    """
    if table_format not in {"xlsx", "csv"}:
        raise ValueError(f"不支持的表格格式：{table_format}")

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    index_entries: list[dict[str, Any]] = []

    for table in tables:
        sheet_name = _excel_sheet_name(table.sheet)
        path = root / f"{table.table_id}.{table_format}"
        _write_table(
            table.frame,
            path,
            table_format=table_format,
            sheet_name=sheet_name,
        )

        stored = _read_table(path, table_format, sheet_name)
        index_entries.append(
            {
                "table_id": table.table_id,
                "domain": table.domain,
                "sheet": table.sheet,
                "sheet_name": sheet_name,
                "format": table_format,
                "path": path.name,
                "rows": int(len(stored)),
                "columns": [
                    {
                        # kind 记录生成时的逻辑类型，用于 CSV 回读时还原日期等类型。
                        "name": str(name),
                        "kind": column_kind(table.frame[name]),
                        "stored_kind": column_kind(stored[name]),
                    }
                    for name in table.frame.columns
                ],
                "source": SOURCE,
            }
        )

    index_path = root / "index.json"
    index_path.write_text(
        json.dumps({"tables": index_entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return index_path


def load_tables(directory: str | Path) -> list[SyntheticTable]:
    root = Path(directory)
    index_path = root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"找不到表格索引：{index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries = payload.get("tables", [])
    if not isinstance(entries, list):
        raise ValueError("表格索引格式无效：tables 必须是数组。")

    tables: list[SyntheticTable] = []
    for entry in entries:
        table_format = entry.get("format", "csv")
        sheet_name = entry.get("sheet_name") or _excel_sheet_name(entry.get("sheet", "工作表"))
        frame = _read_table(root / entry["path"], table_format, sheet_name)
        if table_format == "csv":
            # CSV 不保留类型，按索引里的 datetime 标记还原。
            for column in entry.get("columns", []):
                if column.get("kind") == "datetime" and column["name"] in frame.columns:
                    frame[column["name"]] = pd.to_datetime(
                        frame[column["name"]], errors="coerce"
                    )
        tables.append(
            SyntheticTable(
                table_id=entry["table_id"],
                domain=entry["domain"],
                sheet=entry.get("sheet", "工作表"),
                frame=frame,
            )
        )
    return tables
