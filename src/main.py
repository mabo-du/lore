import sys
from pathlib import Path

# Ensure src is in PYTHONPATH when running directly
sys.path.insert(0, str(Path(__file__).parent))

from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Configure global app settings
    app.setApplicationName("Lore")
    app.setOrganizationName("Digital Heritage Lab")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
