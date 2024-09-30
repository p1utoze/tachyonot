import sys
from .models.llama import SimaticLLM
from .models.whipser import VoiceTranscriber
from .utils.config import whipser_path
from typing import List, Dict, Iterator, Union
from pathlib import Path
from enum import Enum
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLineEdit,
    QScrollArea,
    QFrame,
    QFileDialog,
    QStatusBar,
    QAction
)
import os
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QSize, QTimer, Qt, QUrl, PYQT_VERSION
from PyQt5.QtGui import QIcon, QTextCursor, QFont
from PyQt5.QtMultimedia import QAudioRecorder, QAudioEncoderSettings, QMultimedia

# TODO: Set custom icons pack: "DONE"
# TODO: Optimise the chat streaming (reduce mtimer or disable OpenBLAS): "DONE"
# TODO: Set Audio based command input: "DONE"
# TODO: Create makeself installer from script: "DONE"


ICONS_DIR = Path(__file__).parent / "resources" / "icons"

class DialogMode(Enum):
    ExistingFile = QFileDialog.ExistingFile
    ExistingFolder = QFileDialog.DirectoryOnly


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
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)

        self.sender = QTextEdit()
        f = self.sender.font()
        f.setPointSize(14)
        self.sender.setFont(f)
        self.sender.setReadOnly(True)
        self.sender.setMaximumHeight(30)
        self.sender.setFrameStyle(QFrame.NoFrame)

        self.message = QTextEdit()
        f = self.message.font()
        f.setPointSize(14)
        self.message.setFont(f)
        self.message.setReadOnly(True)
        self.message.setFrameStyle(QFrame.NoFrame)

        self._text_generator = None

        if sender.lower() == "user":
            self.message.setMaximumHeight(30)
            self.setStyleSheet("background-color: #E6F3FF;")
        else:
            self.timer = QTimer()
            self.timer.timeout.connect(self.append_message)
            self.timer.start(1)
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
            self.message.moveCursor(QTextCursor.End)
            self.message.insertPlainText(next_token)
            self.message.moveCursor(QTextCursor.End)
        except StopIteration:
            self.timer.stop()
            del self._text_generator


class ChatWidget(QWidget):
    """
    Custom PySide6 Widget for Chatbot like User Interface.
    """

    message_sent = pyqtSignal(str)
    voice_recording_finished = pyqtSignal(str)

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
        self.input_area.setFocusPolicy(Qt.StrongFocus)
        f = self.input_area.font()
        f.setPointSize(14)  # sets the size to 27
        self.input_area.setFont(f)
        self.send_button = QPushButton(icon=QIcon(str(ICONS_DIR / "send.svg")))
        self.send_button.setIconSize(QSize(35, 35))
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
        self.voice_button = QPushButton(icon=QIcon(str(ICONS_DIR / "microphone.svg")))
        self.voice_button.setToolTip("Record voice")
        self.voice_button.setIconSize(QSize(30, 30))
        self.voice_button.setStyleSheet("background-color: none; border-radius: 20px; border: green;")
        self.voice_button.clicked.connect(self.toggle_voice_recording)
        self.input_layout.addWidget(self.voice_button)

        # File upload button
        # plus_icon = self.style().standardIcon(
        #     QStyle.StandardPixmap.SP_FileDialogNewFolder
        # )
        self.upload_button = QPushButton(icon=QIcon(str(ICONS_DIR / "plus-file.svg")))
        self.upload_button.setToolTip("Attach File")
        self.upload_button.setIconSize(QSize(30, 30))
        self.upload_button.setStyleSheet("background-color: none;")
        self.input_layout.addWidget(self.upload_button)

        self.audio_recorder = QAudioRecorder()
        self.is_recording = False
        self.setup_audio_recorder()
        # print(self.audio_recorder.supportedAudioCodecs())
        # print(self.audio_recorder.supportedContainers())

        self.temp_file = None

    def setup_audio_recorder(self):
        settings = QAudioEncoderSettings()
        settings.setCodec("audio/x-raw")
        settings.setSampleRate(16000)
        settings.setChannelCount(1)
        settings.setQuality(QMultimedia.EncodingQuality.HighQuality)

        self.audio_recorder.setEncodingSettings(settings)
        self.audio_recorder.setContainerFormat("audio/x-wav")

    def toggle_voice_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.voice_button.setStyleSheet("background-color: red; border-radius: 17%")

        # Set up a temporary file for recording
        self.temp_file = os.path.join(os.path.dirname(__file__), "temp_audio.wav")
        self.audio_recorder.setOutputLocation(QUrl.fromLocalFile(self.temp_file))

        self.audio_recorder.record()

    def stop_recording(self):
        self.is_recording = False
        self.voice_button.setStyleSheet("background-color: none;")
        self.audio_recorder.stop()
        self.voice_recording_finished.emit(self.temp_file)

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
        message_box[sender] = message
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message_box)
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )


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
            self._messages = self.messages[-2 * self.k :]

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
        self.voice_assistant = VoiceTranscriber(
            model="tiny.en", models_dir=whipser_path
        )
        self.setWindowTitle("AI Chat Assistant")
        self.setGeometry(100, 100, 1280, 720)

        # Define styling for the QT Application
        self.setStyleSheet(
            """
                    QMainWindow, QWidget { background-color: #f0f0f0; }
                    QTextEdit, QLineEdit { background-color: white; border: 1px solid #ddd; color: black }
                    QPushButton { background-color: #4CAF50; color: white; border: none; padding: 5px; }
                    QPushButton:hover { background-color: #45a049; }
                    QFileDialog { color: white; border: none; padding: 5px; }
                """
        )
        self.processing_style = (
            "QStatusBar { background-color: #FFF3CD; color: #856404; font-weight: bold; }"
            "QStatusBar::item { border: none; }"
        )
        self.finished_style = (
            "QStatusBar { background-color: #03DD4E; color: #155724; font-weight: bold; }"
            "QStatusBar::item { border: none; }"
            "QMenu {color: black; }"
        )
        self.default_style = (
            "QStatusBar { background-color: none; color: black; font-weight: bold; font}"
            "QStatusBar::item { border: none; }"
        )

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.chat_widget = ChatWidget()
        self.layout.addWidget(self.chat_widget)

        self.chat_widget.message_sent.connect(self.handle_user_message)
        self.chat_widget.voice_recording_finished.connect(self.toggle_voice_chat)
        self.chat_widget.upload_button.clicked.connect(self.open_dialog)

        # Initialize conversation memory
        self.memory = ConversationBufferWindowMemory(k=5)

        # Initialize Status Bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(self.default_style)
        self.status_bar.setFont(QFont("Arial", 11))
        self.setStatusBar(self.status_bar)

        button_action = QAction("Built-in RAG Mode", self)
        button_action.setToolTip("Enable or disable context retrieval")
        button_action.setCheckable(True)
        button_action.setChecked(True)
        button_action.toggled.connect(self.toggle_rag_mode)


        menu = self.menuBar()
        menu.setStyleSheet("""
            QMenuBar {color: black; }
            QMenu {color: black; border-radius: 5px; border: 1px solid #000; }
            """
        )

        self.options_menu = menu.addMenu("Options")
        self.options_menu.setStyleSheet(f"background-color: lightblue")
        self.options_menu.addAction(button_action)

        self.file_mode = DialogMode.ExistingFolder.value

    def toggle_rag_mode(self, state):
        val = "lightblue" if state else "#f0f0f0"
        self.file_mode = DialogMode.ExistingFolder.value if state else DialogMode.ExistingFile.value
        state = "Enabled" if state else "Disabled"
        self.options_menu.setStyleSheet(f"background-color: {val}")
        self.status_bar.showMessage(f"RAG Mode: {state}", 2000)

    @pyqtSlot(str)
    def handle_user_message(self, message):
        """
        This slot method adds user-assistant conversations to the chat area
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
        self.chat_widget.add_message(
            "Assistant", self.chat_assistant.invoke(query, stream=True)
        )

    def toggle_voice_chat(self, audio_file):
        # Implement voice chat functionality here
        print("Voice chat toggled")
        transcription = self.voice_assistant.transcribe(audio_file)
        # Add the transcribed text to the chat
        self.chat_widget.input_area.setText(transcription[0].text)

        # Clean up the temporary audio file
        os.remove(audio_file)

    def open_dialog(self):
        """
        A slot method triggered on clicking the
        'attach document' button, opens a file explorer
        :return:
        """
        dialog = QFileDialog(self, caption="Select a directory")
        dialog.setStyleSheet(
            """
        QWidget { font-size: 14px; color: black; selection-color: #4CAF50 }
        QWidget::item { selection-color: #4CAF50 }
        """
        )
        dialog.setFileMode(self.file_mode if not None else DialogMode.ExistingFile)
        if dialog.exec_():  # Start the dialog box
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
            self.status_bar.showMessage("Processing the document. Please wait...")
            self.chat_widget.upload_button.setEnabled(False)
            QApplication.processEvents()  # Collect signal from blocking events

            self.chat_assistant.store_documents(file_path=str(file_path))
            self.status_bar.setStyleSheet(self.finished_style)
            self.status_bar.showMessage("Document processed successfully!")
            self.chat_widget.upload_button.setEnabled(True)

        except BaseException as e:
            print(e)
            self.status_bar.setStyleSheet("background-color: red")
            self.status_bar.showMessage(
                "Error processing documents: " + str(e), msecs=10000
            )

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
