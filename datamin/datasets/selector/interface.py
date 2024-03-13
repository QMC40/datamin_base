from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    Footer,
    Button,
    DataTable,
    SelectionList,
    RadioButton,
    RadioSet,
    Input,
    Label,
    Rule,
)
from textual.screen import Screen
from textual.containers import Horizontal, Vertical, Grid, ScrollableContainer


class QuitScreen(Screen):
    """Screen with a dialog to quit."""

    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Read to save and proceed?", id="question"),
            Button("Save", variant="error", id="quit"),
            Button("Cancel", variant="primary", id="cancel"),
            id="dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.app.exit()
        else:
            self.app.pop_screen()


class FeatureSelectionApp(App):
    CSS_PATH = "interface.tcss"

    BINDINGS = [("q", "request_quit", "Save & Proceed"), ("r", "reset", "Reset")]

    def __init__(
        self,
        feature_names: list[str],
        features: list[str],
        categorical: list[int],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_names = feature_names
        self.features = features
        self.categorical = categorical

        self.reset = False

    def compose(self) -> ComposeResult:
        self.header = Header("Feature Selection")

        print(
            *list(
                zip(
                    self.feature_names,
                    range(len(self.feature_names)),
                    [False] * len(self.feature_names),
                ),
            )
        )

        self.selection_list = SelectionList[int](
            *list(
                zip(
                    self.feature_names,
                    range(len(self.feature_names)),
                    [False] * len(self.feature_names),
                ),
            )
        )

        self.categorical_list = SelectionList[int](
            *list(
                zip(
                    self.feature_names,
                    range(len(self.feature_names)),
                    [
                        True if i in self.categorical else False
                        for i in range(len(self.feature_names))
                    ],
                ),
            )
        )

        self.target_name = Input(
            placeholder=f"{self.feature_names[-1]}",
            id="target_name",
        )

        self.table = DataTable()
        self.footer = Footer()

        with Vertical():
            yield self.header
            with Horizontal():
                with Vertical(id="nan_behaviour"):
                    yield Label("Set Target Attribute (Should be binary 0/1)")
                    yield self.target_name
                    yield Label("Specify NaN Behaviour")
                    with RadioSet(id="nan_behaviour"):
                        yield RadioButton("Remove Row", value="remove_row")
                        yield RadioButton("Remove Column", value="remove_column")
                        yield RadioButton("Replace Mean", value="replace_mean")
                        yield RadioButton("Replace Median", value="replace_median")

                yield Rule(orientation="vertical")
                with Vertical(id="options"):
                    yield Label("Select Protected Features")
                    with ScrollableContainer(
                        id="selection_list",
                    ):
                        yield self.selection_list
                    yield Rule()
                    yield Label("Set Categorical Features")
                    with ScrollableContainer():
                        yield self.categorical_list
                yield Rule(orientation="vertical")
                with Vertical(id="data_preview"):
                    yield Label("Data Preview (100 rows)")
                    yield self.table
            yield self.footer

    def on_mount(self) -> None:
        self.selection_list.border_title = "Select Features"
        self.selection_list.tooltip = (
            "Select the features to be protected through minimization."
        )

        self.categorical_list.border_title = "Categorical Features"
        self.categorical_list.tooltip = (
            "Select the which features should be treated as categorical variables."
        )

        self.nan_selector = self.query_one(RadioSet)

        self.table.add_columns(*self.feature_names)
        self.table.add_rows(self.features)

        self.nan_selector.value = "remove_row"
        self.nan_selector.tooltip = "Select the behaviour for NaN values"
        self.nan_selector.border_title = "NaN Behaviour"

    def action_request_quit(self) -> None:
        self.push_screen(QuitScreen())

    def action_reset(self) -> None:
        self.reset = True
        self.exit()
