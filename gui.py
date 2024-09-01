import tachyonot
from sys import argv, exit

if __name__ == "__main__":
    app = tachyonot.QApplication(argv)
    window = tachyonot.MainWindow()
    window.show()
    exit(app.exec())