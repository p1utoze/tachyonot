import sys
from .models.llama import SimaticLLM
from .models.whipser import VoiceTranscriber
from .utils.config import whipser_path
from typing import List, Dict, Iterator, Union
from pathlib import Path
from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
                               QPushButton, QLineEdit, QScrollArea, QFrame, QStyle, QFileDialog, QStatusBar)
from PySide2.QtCore import  Signal, Slot, QSize, QTimer, Qt
from PySide2.QtGui import QIcon, QTextCursor
import nltk
import os


class ChatBox(QFrame):
    """A Chat UI for each User-Assistant Conversation"""
    def __init__(self, sender: str, parent=None):
        """
        Initialize the Chat UI for the particular role.
        ---
        :param sender: Type of Chat-based LLM Role. [User, Assistant]
        :param parent: Parent window of the Chat display frame widget

        :raises: RuntimeError if the sender argument is not in [user, assistant]
        :return:  A QFrame widget instance

        """
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)

        self.sender = QTextEdit()
        self.sender.setReadOnly(True)
        self.sender.setMaximumHeight(30)
        self.sender.setFrameStyle(QFrame.NoFrame)

        self.message = QTextEdit()
        self.message.setReadOnly(True)
        self.message.setFrameStyle(QFrame.NoFrame)

        self._text_generator = None

        if sender.lower() == "user":
            self.message.setMaximumHeight(30)
            self.setStyleSheet("background-color: #E6F3FF;")
        else:
            self.timer = QTimer()
            self.timer.timeout.connect(self.append_message)
            self.timer.start(50)
            self.setStyleSheet("background-color: #F0F0F0;")

        layout.addWidget(self.sender)
        layout.addWidget(self.message)

    def __setitem__(self, sender, message):
        """
        Set the content of the ChatBox Frame
        For User, directly sets the input prompt
        For assistant, Initializes a Pyside6 Clock for Non-Blocking text stream
        :param sender:
        :param message:
        :return:
        """
        self.sender.setText(f"<b>{sender}</b>")
        if sender.lower() == "user":
            self.message.setPlainText(message)
        else:
            self._text_generator = message

    def append_message(self):
        """
        PySide Slot method triggered due to timeout signal of QTimer

        :return: generates non-blocking, streaming text
        """
        if self._text_generator is None:
            return
        try:
            next_token = next(self._text_generator)
            self.message.moveCursor(QTextCursor.MoveOperation.End)
            self.message.insertPlainText(next_token)
            self.message.moveCursor(QTextCursor.MoveOperation.End)
        except StopIteration:
            self.timer.stop()
            del self._text_generator


class ChatWidget(QWidget):
    """
    Custom PySide6 Widget for Chatbot like User Interface.
    """
    message_sent = Signal(str)

    def __init__(self):
        """
        Initialize the Chat UI.
        Following child widgets from QtWidgets module are initialized:
        [QScrollArea,
        QWidget,
        QLineEdit,
        QPushButton]
        """
        super().__init__()
        self.layout = QVBoxLayout(self)

        # Text Area
        self.chat_area = QScrollArea()
        self.chat_area.setWidgetResizable(True)
        self.chat_content = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_content)
        self.chat_layout.addStretch()
        self.chat_area.setWidget(self.chat_content)

        # Input Box
        self.input_area = QLineEdit()
        self.input_area.setPlaceholderText("What's on your mind today?")
        self.input_area.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.send_button = QPushButton(icon=QIcon.fromTheme("mail-send"))
        self.send_button.setIconSize(QSize(30, 30))
        self.send_button.setStyleSheet("background-color: none;")
        self.send_button.setToolTip("Send")

        self.layout.addWidget(self.chat_area)

        # Add input text and send button to layout
        self.input_layout = QHBoxLayout()
        self.input_layout.addWidget(self.input_area)
        self.input_layout.addWidget(self.send_button)
        self.input_layout.setSpacing(-50)
        self.layout.addLayout(self.input_layout)

        # Connect signal of clicked and return key to appropriate slots
        self.send_button.clicked.connect(self.send_message)
        self.input_area.returnPressed.connect(self.send_message)

        # Voice chat button
        self.voice_button = QPushButton(icon=QIcon.fromTheme("audio-input-microphone"))
        self.voice_button.setToolTip("Record voice")
        self.voice_button.setIconSize(QSize(25, 25))
        self.voice_button.setStyleSheet("background-color: none;")
        self.input_layout.addWidget(self.voice_button)

        # File upload button
        plus_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder)
        self.upload_button = QPushButton(icon=plus_icon)
        self.upload_button.setToolTip("Attach File")
        self.upload_button.setIconSize(QSize(20, 20))
        self.upload_button.setStyleSheet("background-color: none;")
        self.input_layout.addWidget(self.upload_button)


    def send_message(self):
        """
        Slot method that receives the prompt and resets the textbox
        :return: None
        """
        message = self.input_area.text().strip()
        if message:
            self.message_sent.emit(message)
            self.input_area.clear()

    def add_message(self, sender, message: Union[str, Iterator]):
        """
        Insert a New conversation for each prompt
        :param sender: Frame type. ["user", "assistant"]
        :param message: A string or a generator object from LLM
        :return:
        """
        message_box = ChatBox(sender=sender)
        # print("Invoked")
        message_box[sender] = message
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message_box)
        self.chat_area.verticalScrollBar().setValue(self.chat_area.verticalScrollBar().maximum())


class ConversationBufferWindowMemory:
    """
    A Chat history implementation based on K-Windows Buffer from Langchain
    """
    def __init__(self, k: int = 5):
        """
        Initializet the chat memory.
        :param k: Tells k number of conversations to remember for LLM
        """
        self.k = k
        self._messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > 2 * self.k:
            self._messages = self.messages[-2 * self.k:]

    @property
    def messages(self) -> List[Dict[str, str]]:
        return self._messages

    def get_context(self) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages])



class MainWindow(QMainWindow):
    """
    Main PyQT window launched with a AI chatbot Interface
    """
    def __init__(self):
        super().__init__()
        self.chat_assistant = SimaticLLM()  # Initialize your custom chat assistant
        self.voice_assistant = VoiceTranscriber(model="tiny.en", models_dir=whipser_path)
        self.setWindowTitle("AI Chat Assistant")
        self.setGeometry(100, 100, 800, 600)

        # Define styling for the QT Application
        self.setStyleSheet("""
                    QMainWindow, QWidget { background-color: #f0f0f0; }
                    QTextEdit, QLineEdit { background-color: white; border: 1px solid #ddd; color: black }
                    QPushButton { background-color: #4CAF50; color: white; border: none; padding: 5px; }
                    QPushButton:hover { background-color: #45a049; }
                    QFileDialog { color: white; border: none; padding: 5px; }
                """)
        self.processing_style = (
            "QStatusBar { background-color: #FFF3CD; color: #856404; font-weight: bold; }"
            "QStatusBar::item { border: none; }"
        )
        self.finished_style = (
            "QStatusBar { background-color: #03DD4E; color: #155724; font-weight: bold; }"
            "QStatusBar::item { border: none; }"
        )
        self.default_style = (
            "QStatusBar { background-color: none; color: black; }"
            "QStatusBar::item { border: none; }"
        )

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.chat_widget = ChatWidget()
        self.layout.addWidget(self.chat_widget)

        self.chat_widget.message_sent.connect(self.handle_user_message)
        self.chat_widget.voice_button.clicked.connect(self.toggle_voice_chat)
        self.chat_widget.upload_button.clicked.connect(self.open_dialog)

        # Initialize conversation memory
        self.memory = ConversationBufferWindowMemory(k=5)

        # Initialize Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    @Slot(str)
    def handle_user_message(self, message):
        """

        :param message:
        :return:
        """
        self.chat_widget.add_message("User", message)
        self.memory.add_message("user", message)
        self.generate_response(message)

    def generate_response(self, query):
        """
        Invokes the llm by passing a generator object for streaming
        :param query: The user prompt
        :return:
        """
        # context = self.memory.get_context()
        self.chat_widget.add_message("Assistant", self.chat_assistant.invoke(query, stream=True))

    def toggle_voice_chat(self):
        # Implement voice chat functionality here
        print("Voice chat toggled")

    def open_dialog(self):
        """
        A slot method triggered on clicking the
        'attach document' button, opens a file explorer
        :return:
        """
        dialog = QFileDialog(self, caption="Select a directory")
        dialog.setStyleSheet("""
        QWidget { font-size: 14px; color: black; selection-color: #4CAF50 }
        QWidget::item { selection-color: #4CAF50 }
        """)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if dialog.exec_():       # Start the dialog box
            file_name = dialog.selectedUrls()[0]
            if file_name:
                path = Path(file_name.path())
                self.create_file_embeds(path)

        dialog.destroy()
        QTimer.singleShot(5000, self.reset_status_bar)

    def create_file_embeds(self, file_path: Path) -> None:
        """
        Create embeddings from the file and add the documents to Faiss vector store.
        :param file_path: Path to the file selected from the file menu
        :return:
        """
        try:
            self.status_bar.setStyleSheet(self.processing_style)
            self.status_bar.showMessage("Processing the document.Please wait...")
            self.chat_widget.upload_button.setEnabled(False)
            QApplication.processEvents() # Collect signal from blocking events

            self.chat_assistant.store_documents(file_path=str(file_path))
            self.status_bar.setStyleSheet(self.finished_style)
            self.status_bar.showMessage("Document processed successfully!")
            self.chat_widget.upload_button.setEnabled(True)

        except BaseException as e:
            print(e)
            self.status_bar.setStyleSheet("background-color: red")
            self.status_bar.showMessage("Error processing documents: " + str(e), timeout=10000)

    def reset_status_bar(self):
        self.status_bar.setStyleSheet(self.default_style)
        self.status_bar.clearMessage()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()