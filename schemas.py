from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# 定义聊天请求类，设定需要的内容
class ChatRequest(BaseModel):
    session_id: str
    question: str


# Excel 插件：把当前工作表数据（表头 + 数据行）上传以建立会话
class SessionRequest(BaseModel):
    columns: list[str]
    rows: list[list[Any]]


class FilterCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    operator: Literal[
        "eq", "ne", "gt", "gte", "lt", "lte", "contains",
        "not_contains", "is_empty", "not_empty",
    ]
    value: Any = None


class Metric(BaseModel):
    """单个统计指标。

    ``count`` 的语义是**该组有多少行记录**，与 ``column`` 是否含空值无关
    （等价于按组计数，不是 ``Series.count``）。如需统计某列的非空值个数，
    请先用 ``filter_rows`` 过滤空值。

    ``nunique`` 统计 ``column`` 中不同非空值的个数。
    """

    model_config = ConfigDict(extra="forbid")

    column: str
    function: Literal["sum", "mean", "min", "max", "count", "nunique"]
    name: str | None = None


class SortKey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    ascending: bool = True


class FilterRowsInput(BaseModel):
    filters: list[FilterCondition] = Field(
        min_length=1,
        description="需要同时满足的筛选条件",
    )


class SortRowsInput(BaseModel):
    sort_by: list[SortKey] = Field(min_length=1, description="排序字段和顺序")


class AggregateDataInput(BaseModel):
    group_by: list[str] = Field(default_factory=list, description="可选的分组字段")
    metrics: list[Metric] = Field(min_length=1, description="统计字段和统计方式")


class TopNRowsInput(BaseModel):
    sort_by: list[SortKey] = Field(min_length=1, description="决定前 N 名的字段")
    limit: int = Field(ge=1, le=10000, description="返回行数")


class DeduplicateRowsInput(BaseModel):
    subset: list[str] = Field(
        default_factory=list,
        description="用于判重的字段；留空表示按整行判重",
    )


class FillMissingInput(BaseModel):
    fill_strategy: Literal["ffill", "bfill", "value"]
    subset: list[str] = Field(default_factory=list, description="要填充的字段；留空表示全部字段")
    fill_value: Any = None

    @model_validator(mode="after")
    def validate_fill_value(self) -> "FillMissingInput":
        if self.fill_strategy == "value":
            if isinstance(self.fill_value, (dict, list, set, tuple)) or self.fill_value is None:
                raise ValueError("value 填充策略需要一个非空标量 fill_value")
        return self


class ComputedColumnInput(BaseModel):
    name: str = Field(min_length=1, description="新列名称，不能与现有字段重名")
    operator: Literal["add", "subtract", "multiply", "divide"]
    left: str = Field(min_length=1, description="左侧数值字段")
    right_column: str | None = Field(default=None, description="右侧数值字段")
    right_value: float | None = Field(default=None, description="右侧常数")
    round_digits: int | None = Field(
        default=None,
        ge=0,
        le=10,
        description="结果保留的小数位数；留空表示不额外取整",
    )

    @field_validator("name", "left", "right_column")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("字段名称不能为空")
        return stripped

    @model_validator(mode="after")
    def validate_operands(self) -> "ComputedColumnInput":
        if (self.right_column is None) == (self.right_value is None):
            raise ValueError("right_column 和 right_value 必须且只能提供一个")
        return self


class CreateChartInput(BaseModel):
    chart_type: Literal["bar", "line", "pie", "scatter"]
    chart_y: str = Field(description="数值字段")
    chart_x: str | None = Field(default=None, description="类别或横轴字段")
    chart_title: str | None = None

    @model_validator(mode="after")
    def validate_scatter_axes(self) -> "CreateChartInput":
        if self.chart_type == "scatter" and not self.chart_x:
            raise ValueError("散点图需要 chart_x 字段")
        return self
