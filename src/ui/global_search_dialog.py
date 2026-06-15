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
from PyQt6.QtCore import QThread, pyqtSignal
from lore_core.global_search import GlobalSearchIndex
import datetime


class SearchWorker(QThread):
    """Runs search in a background thread to avoid UI freeze."""

    finished = pyqtSignal(list)  # results list
    error = pyqtSignal(str)

    def __init__(self, search_index, query: str, use_semantic: bool, parent=None):
        super().__init__(parent)
        self.search_index = search_index
        self.query = query
        self.use_semantic = use_semantic

    def run(self):
        try:
            if self.use_semantic:
                results = self.search_index.search_semantic(self.query)
            else:
                results = self.search_index.search_keyword(self.query)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


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

        # Lazy-loaded: init in perform_search to avoid blocking UI on model load
        self.search_index = None

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

    def _populate_results(self, results):
        """Populate the results list from search results."""
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

        self.btn_search.setEnabled(True)
        self.search_input.setEnabled(True)

    def perform_search(self):
        query = self.search_input.text().strip()
        if not query:
            return

        self.results_list.clear()
        self.btn_search.setEnabled(False)
        self.search_input.setEnabled(False)
        self.results_list.addItem("Searching...")

        # Lazy-load the search index on first search (avoid UI freeze on dialog open)
        if self.search_index is None:
            try:
                self.search_index = GlobalSearchIndex()
            except Exception as e:
                self.results_list.clear()
                self.results_list.addItem(f"Failed to initialise search engine: {e}")
                self.btn_search.setEnabled(True)
                self.search_input.setEnabled(True)
                return

        # Run search in background thread to avoid UI freeze during vector inference
        use_semantic = self.radio_semantic.isChecked()
        self.search_worker = SearchWorker(self.search_index, query, use_semantic)
        self.search_worker.finished.connect(self._populate_results)
        self.search_worker.error.connect(self._on_search_error)
        self.search_worker.start()

    def _on_search_error(self, err_msg: str):
        self.results_list.addItem(f"Search failed: {err_msg}")
        self.btn_search.setEnabled(True)
        self.search_input.setEnabled(True)
