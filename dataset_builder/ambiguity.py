"""检测表格中真实存在的歧义，并生成对应的澄清说法。

设计原则：歧义必须**可验证**——两个候选选项在数据上真的会给出不同结果，
否则"澄清"只是多余的打扰。例如「订单数」只有在去重数 ≠ 行数时才真正歧义；
若订单号恰好唯一，两种理解结果相同，就不该反问。

覆盖的歧义类型：
1. ``count_semantics``  计数口径：记录条数 vs 去重编号个数
2. ``metric_column``    指标所指：表里两个数值列都可能被"金额/时长/库存"指代
3. ``aggregation``      统计方式：量表列求和 vs 求平均
4. ``group_dimension``  分组维度：多个分类列都可能被"按分类"指代
5. ``time_range``       时间范围：日期跨多个月，"最近/今年"边界不清
6. ``missing_value``    缺失值处理：数值列有空值时，平均值是否忽略空行
7. ``top_n``            条数未指定："最多的几个"
8. ``chart_type``       图表类型未指定："画个图看看分布"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from dataset_builder.quality import is_identifier_column
from dataset_builder.tables import SyntheticTable, column_kind

RATING_HINTS = ("满意度", "评分", "打分", "星级", "推荐意愿", "得分", "评分值")

# 每个业务域里最容易被整体指代的数值列组：模糊说法 -> 候选列。
VAGUE_METRIC_TERMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "sales_orders": ("金额", ("单价", "销售额")),
    "inventory": ("库存", ("库存数量", "安全库存")),
    "staffing": ("时长", ("工时", "加班时长")),
    "budget": ("支出", ("实际支出", "差异")),
    "survey": ("意愿", ("满意度", "推荐意愿")),
    # 留出域：与训练集词汇不重叠，用于检验对新表结构的迁移。
    # 只列出确实有两个可比数值列的域；inspection 只有一个数值列，不构成指标歧义。
    "logistics": ("运费", ("运费", "单位运费")),
    "customers": ("金额", ("签约金额", "回款金额")),
}


@dataclass
class Ambiguity:
    """一个可验证的歧义场景。"""

    kind: str
    vague_term: str
    options: tuple[str, str]
    # 澄清模板，使用 {term} / {a} / {b} 占位。
    clarifications: tuple[str, ...]
    evidence: dict[str, Any] = field(default_factory=dict)
    # 提问里允许出现的模糊说法；任一命中即认为聚焦到了本歧义。
    accept_terms: tuple[str, ...] = ()
    # 对用户展示的"歧义说明"，供 LLM 理解该问什么。
    hint: str = ""

    def render(self, index: int = 0) -> str:
        template = self.clarifications[index % len(self.clarifications)]
        # evidence 里的字段（如 missing 数量）也作为占位符来源。
        return template.format(
            term=self.vague_term,
            a=self.options[0],
            b=self.options[1],
            **self.evidence,
        )

    def matches(self, question: str) -> bool:
        terms = self.accept_terms or (self.vague_term,)
        return any(term and term in question for term in terms)


# 每种歧义的澄清措辞：只问缺的那一点，不复述已知条件。
CLARIFY_TEMPLATES: dict[str, tuple[str, ...]] = {
    "count_semantics": (
        "想确认一下：「{term}」是按记录条数算，还是按去重后的「{a}」个数算？有重复记录时两者不同。",
        "「{term}」有两种算法：数记录行数，或数不重复的「{a}」。您要哪一种？",
        "请问「{term}」要统计总行数，还是「{a}」去重后的数量？",
    ),
    "metric_column": (
        "「{term}」是指「{a}」还是「{b}」？两者含义不同，请确认。",
        "想确认指标：「{term}」按「{a}」算还是按「{b}」算？",
        "这里需要确认口径：「{term}」用「{a}」还是「{b}」？",
    ),
    "aggregation": (
        "「{a}」要算合计还是平均值？这类评分通常看平均值，但想跟您确认。",
        "想确认统计方式：「{a}」是求和还是求平均？",
        "「{a}」按合计算还是按均值算？请确认。",
    ),
    "group_dimension": (
        "「按分类」是指按「{a}」分组，还是按「{b}」分组？",
        "想确认分组维度：用「{a}」还是「{b}」？",
        "请确认按哪一个分类汇总：「{a}」还是「{b}」？",
    ),
    "time_range": (
        "数据里日期从 {a} 到 {b}，「最近」具体指哪一段？例如最近一个月还是全部？",
        "想确认时间范围：是要全部 {a}～{b} 的数据，还是只看其中某一段？",
        "「最近」的边界请确认一下：日期范围是 {a}～{b}，您要哪一段？",
    ),
    "missing_value": (
        "「{a}」有 {missing} 个空值，计算时要把空白当 0，还是直接跳过这些行？",
        "「{a}」存在 {missing} 条空记录，平均值是忽略空行还是把空值算作 0？",
        "先确认一下：「{a}」有 {missing} 个空值，统计时排除它们还是按 0 处理？",
    ),
    "top_n": (
        "要取前几条？请给个具体数量，我再按您关心的指标排序。",
        "「{term}」具体要几名？麻烦确认个数。",
        "想确认取多少条：排序后取前几？",
    ),
    "chart_type": (
        "想用哪种图？柱状图适合比大小，饼图看占比，折线图看趋势。",
        "「{a}」画成柱状图、饼图还是折线图？请确认一下。",
        "请确认图表类型：柱状图 / 饼图 / 折线图？",
    ),
}


def _populated(series: pd.Series) -> bool:
    return bool(series.notna().any())


def _numeric_columns(table: SyntheticTable) -> list[str]:
    return [
        str(c) for c in table.frame.columns
        if column_kind(table.frame[c]) == "numeric"
        and _populated(table.frame[c])
        and not is_identifier_column(str(c))
    ]


def _categorical_columns(table: SyntheticTable) -> list[str]:
    return [
        str(c) for c in table.frame.columns
        if column_kind(table.frame[c]) == "text"
        and not is_identifier_column(str(c))
        and 1 < table.frame[c].nunique() <= 12
    ]


def _identifier_column(table: SyntheticTable) -> str | None:
    for column in table.frame.columns:
        if is_identifier_column(str(column)) and _populated(table.frame[column]):
            return str(column)
    return None


# 标识列 -> 用户口中常见的"计数说法"。
COUNT_TERMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "订单号": ("订单数", ("订单数", "多少单", "几单", "订单量")),
    "员工编号": ("员工数", ("员工数", "多少人", "几人", "人数")),
    "受访者编号": ("受访者数", ("受访者数", "多少人", "几个人", "受访人数")),
    "商品编码": ("商品数", ("商品数", "几种商品", "多少个商品", "商品种类")),
}


def detect(table: SyntheticTable) -> list[Ambiguity]:
    """返回该表上真实存在的歧义场景。"""
    found: list[Ambiguity] = []
    frame = table.frame
    rows = len(frame)

    # 1. 计数口径：只有去重数 != 行数时才真的歧义。
    identifier = _identifier_column(table)
    if identifier:
        distinct = int(frame[identifier].nunique(dropna=True))
        if distinct != rows:
            term, accepts = COUNT_TERMS.get(identifier, (f"{identifier}数量", ()))
            found.append(
                Ambiguity(
                    kind="count_semantics",
                    vague_term=term,
                    options=(identifier, "记录条数"),
                    clarifications=CLARIFY_TEMPLATES["count_semantics"],
                    evidence={"rows": rows, "distinct": distinct},
                    accept_terms=accepts or (term,),
                    hint=f"用户说的「{term}」可能指记录条数，也可能指不同的{identifier}个数",
                )
            )

    # 2. 指标所指：同域两个数值列都可能被模糊词指代。
    vague = VAGUE_METRIC_TERMS.get(table.domain)
    if vague:
        term, candidates = vague
        present = [c for c in candidates if c in frame.columns and _populated(frame[c])]
        if len(present) >= 2:
            found.append(
                Ambiguity(
                    kind="metric_column",
                    vague_term=term,
                    options=(present[0], present[1]),
                    clarifications=CLARIFY_TEMPLATES["metric_column"],
                    evidence={"candidates": present},
                    accept_terms=(term, *present),
                    hint=f"用户说的「{term}」可能指{ present[0]}，也可能指{present[1]}",
                )
            )

    # 3. 统计方式：量表列求和 vs 平均，两种口径数值差异显著。
    for column in frame.columns:
        name = str(column)
        if not any(hint in name for hint in RATING_HINTS):
            continue
        if column_kind(frame[column]) != "numeric" or not _populated(frame[column]):
            continue
        found.append(
            Ambiguity(
                kind="aggregation",
                vague_term="汇总",
                options=(name, "平均值"),
                clarifications=CLARIFY_TEMPLATES["aggregation"],
                evidence={"column": name, "rows": rows},
                accept_terms=("汇总", "统计", "看一下", "情况", "表现", "怎么样"),
                hint=f"用户说「汇总{name}」但没说合计还是平均，两种口径差别很大",
            )
        )
        break

    # 4. 分组维度：至少两个分类列。
    categorical = _categorical_columns(table)
    if len(categorical) >= 2:
        found.append(
            Ambiguity(
                kind="group_dimension",
                vague_term="分类",
                options=(categorical[0], categorical[1]),
                clarifications=CLARIFY_TEMPLATES["group_dimension"],
                evidence={"candidates": categorical},
                accept_terms=("分类", "分组", "按类", "各个类", "分门别类"),
                hint=f"用户说「按分类」但表里有{len(categorical)}个分类列可分组",
            )
        )

    # 5. 时间范围：日期跨度超过 30 天。
    for column in frame.columns:
        if column_kind(frame[column]) != "datetime":
            continue
        series = pd.to_datetime(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        span = (series.max() - series.min()).days
        if span > 30:
            found.append(
                Ambiguity(
                    kind="time_range",
                    vague_term="最近",
                    options=(str(series.min().date()), str(series.max().date())),
                    clarifications=CLARIFY_TEMPLATES["time_range"],
                    evidence={"column": str(column), "span_days": span},
                    accept_terms=("最近", "近期", "这段时间", "最新", "近期"),
                    hint=f"用户说「最近」但数据跨 {span} 天，边界不清",
                )
            )
            break

    # 6. 缺失值处理：数值列存在空值。
    for column in _numeric_columns(table):
        missing = int(frame[column].isna().sum())
        if missing:
            found.append(
                Ambiguity(
                    kind="missing_value",
                    vague_term="空值",
                    options=(column, "忽略空值"),
                    clarifications=CLARIFY_TEMPLATES["missing_value"],
                    evidence={"column": column, "missing": missing},
                    accept_terms=("空值", "空白", "没填", "缺失", "空着", "为空"),
                    hint=f"「{column}」有 {missing} 个空值，统计时是否排除未说明",
                )
            )
            break

    # 7. 条数未指定：存在可排序的数值列。
    numeric = _numeric_columns(table)
    if numeric:
        found.append(
            Ambiguity(
                kind="top_n",
                vague_term="最多的几个",
                options=(numeric[0], "排序"),
                clarifications=CLARIFY_TEMPLATES["top_n"],
                evidence={"column": numeric[0]},
                accept_terms=("最多", "最高", "最大", "排前", "前几", "靠前", "排名"),
                hint=f"用户说「最多/前几」但没给具体条数，且表里有{len(numeric)}个数值列可排序",
            )
        )

    # 8. 图表类型未指定：只要表里有数值列。
    if numeric:
        found.append(
            Ambiguity(
                kind="chart_type",
                vague_term="画个图",
                options=(numeric[0], "图表类型"),
                clarifications=CLARIFY_TEMPLATES["chart_type"],
                evidence={"column": numeric[0]},
                accept_terms=("画个图", "画图", "图表", "可视化", "出个图", "图形"),
                hint=f"用户说「画个图」但没说柱状图/饼图/折线图，数据可用{numeric[0]}",
            )
        )

    return found
