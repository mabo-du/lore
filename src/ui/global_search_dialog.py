from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QListWidget,
    QLabel,
    QListWidgetItem,
    QRadioButton,
    QButtonGroup,
    QWidget,
)
from lore_core.global_search import GlobalSearchIndex
import datetime


class GlobalSearchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Global Archive Search")
        self.resize(700, 500)

        self.setStyleSheet("""
            QDialog { background-color: #121212; color: #cccccc; }
            QLineEdit {
                background-color: #2d2d2d;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 6px;
                color: white;
                font-size: 14px;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 6px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0098ff; }
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #333333;
                outline: none;
            }
            QListWidget::item {
                border-bottom: 1px solid #333333;
                padding: 10px;
            }
            QListWidget::item:selected {
                background-color: #2d2d2d;
            }
        """)

        try:
            self.search_index = GlobalSearchIndex()
        except Exception as e:
            self.search_index = None
            print(f"Failed to init search index: {e}")

        layout = QVBoxLayout(self)

        # Search Bar
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search all transcripts...")
        self.search_input.returnPressed.connect(self.perform_search)

        self.btn_search = QPushButton("Search")
        self.btn_search.clicked.connect(self.perform_search)

        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.btn_search)

        layout.addLayout(search_layout)

        # Options
        options_layout = QHBoxLayout()
        self.type_group = QButtonGroup(self)
        self.radio_keyword = QRadioButton("Exact/Keyword (FTS5)")
        self.radio_semantic = QRadioButton("Semantic/Concept (Vector)")
        self.radio_keyword.setChecked(True)
        self.type_group.addButton(self.radio_keyword)
        self.type_group.addButton(self.radio_semantic)
        options_layout.addWidget(self.radio_keyword)
        options_layout.addWidget(self.radio_semantic)
        options_layout.addStretch()
        layout.addLayout(options_layout)

        # Results List
        self.results_list = QListWidget()
        self.results_list.setWordWrap(True)
        layout.addWidget(self.results_list)

    def _format_time(self, ms: int) -> str:
        seconds = ms // 1000
        td = datetime.timedelta(seconds=seconds)
        return str(td)

    def perform_search(self):
        if not self.search_index:
            self.results_list.clear()
            self.results_list.addItem("Search engine not available.")
            return

        query = self.search_input.text().strip()
        if not query:
            return

        self.results_list.clear()
        self.btn_search.setEnabled(False)
        self.search_input.setEnabled(False)

        # In a real app this should be threaded so it doesn't freeze the UI,
        # but for simplicity we run it inline here.
        try:
            if self.radio_keyword.isChecked():
                results = self.search_index.search_keyword(query)
            else:
                results = self.search_index.search_semantic(query)

            if not results:
                self.results_list.addItem("No results found.")
            else:
                for res in results:
                    item = QListWidgetItem()

                    time_str = f"[{self._format_time(res['start_ms'])}]"
                    proj_id = res["project_id"]
                    snippet = res["snippet"]

                    # Create custom widget for rich text display
                    widget = QWidget()
                    w_layout = QVBoxLayout(widget)
                    w_layout.setContentsMargins(5, 5, 5, 5)

                    header = QLabel(
                        f"<b style='color:#007acc;'>{proj_id}</b> <span style='color:gray;'>{time_str}</span>"
                    )
                    body = QLabel(snippet)
                    body.setWordWrap(True)
                    body.setStyleSheet("color: #dddddd; font-size: 13px;")

                    w_layout.addWidget(header)
                    w_layout.addWidget(body)

                    item.setSizeHint(widget.sizeHint())
                    self.results_list.addItem(item)
                    self.results_list.setItemWidget(item, widget)

        except Exception as e:
            self.results_list.addItem(f"Search failed: {e}")
        finally:
            self.btn_search.setEnabled(True)
            self.search_input.setEnabled(True)
