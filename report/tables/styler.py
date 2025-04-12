from typing import Any, Dict, List

import pandas as pd
from pandas.io.formats.style import Styler

from report.tables.highlight_rule import HighlightRule

FORMAT_BOLD = "font-weight:bold;"


class TableStyler:
    def __init__(self, data_frame: pd.DataFrame, columns: List[str], rules: Dict[str, Any]) -> None:
        self._data_frame = data_frame
        self._columns = columns
        self._rules = rules

    def _highlight_max(self, styler: Styler) -> Styler:
        columns_for_highlight = [
            name
            for name, highlight_rule in self._rules.items()
            if highlight_rule is HighlightRule.MAX
        ]
        columns = list(set(self._columns).intersection(set(columns_for_highlight)))
        return styler.highlight_max(subset=columns, props=FORMAT_BOLD)

    def _highlight_min(self, styler: Styler) -> Styler:
        columns_for_highlight = [
            name
            for name, highlight_rule in self._rules.items()
            if highlight_rule is HighlightRule.MIN
        ]
        columns = list(set(self._columns).intersection(set(columns_for_highlight)))
        return styler.highlight_min(subset=columns, props=FORMAT_BOLD)

    def _round_values(self, styler: Styler) -> Styler:
        columns_for_rounding = [
            name
            for name, highlight_rule in self._rules.items()
            if highlight_rule is not HighlightRule.NONE
        ]
        columns = list(set(self._columns).intersection(set(columns_for_rounding)))
        return styler.format(lambda value: f"{value:.2f}", na_rep="N/A", subset=columns)

    def _hide_index(self, styler: Styler) -> Styler:
        return styler.hide()

    def _highlight_index(self, styler: Styler) -> Styler:
        return styler.applymap_index(lambda _: FORMAT_BOLD, axis="columns")

    def style(self) -> Styler:
        frame = self._data_frame[self._columns]
        styler = frame.style

        styler = self._highlight_max(styler)
        styler = self._highlight_min(styler)
        styler = self._hide_index(styler)
        styler = self._highlight_index(styler)
        styler = self._round_values(styler)
        return styler
