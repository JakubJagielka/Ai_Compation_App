import os.path
import sys
import json
import math 
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
import requests
from queue import Queue
from threading import Event
from os import getcwd
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,QComboBox,QCheckBox,
                               QLineEdit, QPushButton, QScrollArea, QLabel, QSystemTrayIcon, QMenu,QMessageBox, QSizePolicy, QSlider)
from PySide6.QtGui import QIcon,QFont,QPixmap,QColor, QPainter, QPainterPath, QPolygon
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Signal,QTimer, QPoint

import Live2D
from DataProcessing import UserData, save_settings, load_settings


ENCRYPTION_KEY = b'\x9a\x1f\x8b\xcd\xef\x01\x23\x45\x67\x89\xab\xcd\xef\x01\x23\x45'

PRIMARY_COLOR = "#007AFF"
ACCENT_COLOR = "#34C759"
TEXT_PRIMARY_COLOR = "#1C1C1E"
BACKGROUND_PRIMARY_COLOR = "#F2F2F7"
BACKGROUND_SECONDARY_COLOR = "#FFFFFF"
SEPARATOR_COLOR = "#C6C6C8"
ERROR_COLOR = "#FF3B30"
CHAT_AI_BUBBLE_BG = "#E5E5EA"
TOP_BAR_CHAT_COLOR = "#FFFFFF"
TOP_BAR_CHAT_BORDER_COLOR = "#D1D1D6"

AVAILABLE_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Portuguese": "pt",
    "Other (might work worst)": "all"
}

class VoiceIndicatorWidget(QWidget):
    def __init__(self, threshold=4000, parent=None):
        super().__init__(parent)
        self.setMinimumSize(80, 24)
        self.level = 0
        self.threshold = threshold
        self.num_bars = 10
        self.max_level = 32767  # Max int16 value

        self.bar_color_inactive = QColor("#A0A0A5")
        self.bar_color_active = QColor(PRIMARY_COLOR)
        self.background_color_default = QColor(BACKGROUND_PRIMARY_COLOR)
        self.background_color_detecting = QColor("#C7F0D2") # Light green for detection

        self.decay_timer = QTimer(self)
        self.decay_timer.setInterval(150) # ms
        self.decay_timer.timeout.connect(self._decay_level)
        self.decay_timer.setSingleShot(True)

    def _decay_level(self):
        """Slot to reset level to 0 when timer fires."""
        self.level = 0
        self.update()

    def set_level(self, level):
        """Update the current audio level and restart the decay timer."""
        self.level = max(0, level)
        self.decay_timer.start() 
        self.update()

    def set_threshold(self, threshold):
        """Update the detection threshold."""
        self.threshold = threshold
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        is_detecting = self.level > self.threshold

        painter.setPen(Qt.NoPen)
        bg_color = self.background_color_detecting if is_detecting else self.background_color_default
        painter.setBrush(bg_color)
        painter.drawRoundedRect(self.rect(), 5, 5)

        if self.level > 0:
            log_level = math.log(self.level)
            log_max = math.log(self.max_level)
            ratio = min(1.0, log_level / log_max)
            lit_bars = int(ratio * self.num_bars)
        else:
            lit_bars = 0
        lit_bars = min(lit_bars, self.num_bars)

        bar_width = (self.width() - (self.num_bars + 1) * 3) / self.num_bars
        bar_spacing = 3
        total_bar_width = (bar_width + bar_spacing) * self.num_bars - bar_spacing
        start_x = (self.width() - total_bar_width) / 2

        for i in range(self.num_bars):
            bar_color = self.bar_color_active if is_detecting else self.bar_color_inactive
            
            if i < lit_bars:
                painter.setBrush(bar_color)
            else:
                painter.setBrush(QColor("#E5E5EA")) 

            x = start_x + i * (bar_width + bar_spacing)
            painter.drawRect(int(x), 4, int(bar_width), self.height() - 8)


def load_characters() -> dict:
    models = {}
    try:
        resources_path = "Resources"
        if not os.path.exists(resources_path):
            print(f"Warning: '{resources_path}' directory not found.")
            return models
        for model_name in os.listdir(resources_path):
            try:
                model_path = os.path.join(resources_path, model_name)
                if os.path.isdir(model_path):
                    cur_files  = os.listdir(model_path)
                    model_files = [file for file in cur_files if file.endswith('.model3.json')]
                    model_file = model_files[0] if model_files else None
                    model_name = '.'.join(model_file.split('.')[:-2])
                    if not any(f.endswith('.moc3') for f in cur_files) or \
                    not any(f.endswith('.model3.json') for f in cur_files):
                        continue
                    
                    model_json_path = os.path.join(model_path, model_file)
                    
                    with open(model_json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    expressions = data.get("FileReferences", {}).get("Expressions", [])
                    motions = data.get("FileReferences", {}).get("Motions", {}).get("Idle", [])
                    system = ""
                    voices = {}
                    
                    system_json_path = os.path.join(model_path, f"{model_name}.system.json")
                    if os.path.exists(system_json_path):
                        with open(system_json_path, 'r', encoding='utf-8') as f:
                            system_file = json.load(f)
                        
                        if isinstance(system_file, dict) and "system" in system_file:
                            system = system_file.get("system", "")
                            voices = system_file.get("voices", {})
                        else:
                            system = system_file
                    
                    if model_name == "Haru":
                        model_name = "Therapist"
                    models[model_name] = {
                        "expressions": expressions, "motions": motions,
                        "system": system, "voices": voices
                    }
            except Exception as e:
                print(f"Error loading model '{model_name}': {e}")
                continue
    except Exception as e:
        print(f"Error in loading models: {e}")
        

    return models

# MODIFIED: ChatMessageWidget has been updated for a more modern design.
class ChatMessageWidget(QWidget):
    def __init__(self, text, is_user, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.user_color = QColor(PRIMARY_COLOR)
        self.ai_color = QColor(CHAT_AI_BUBBLE_BG)
        self.text_color_user = QColor("#FFFFFF")
        self.text_color_ai = QColor(TEXT_PRIMARY_COLOR)
        self.bubble_radius = 20  # Softer corners
        self.tail_width = 12
        self.tail_height = 12

        layout = QVBoxLayout(self)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.PlainText)
        self.label.setStyleSheet(f"""
            QLabel {{
                color: {self.text_color_user.name() if is_user else self.text_color_ai.name()};
                font: 15px 'Segoe UI';
                padding: 0px; /* Let layout margins handle padding */
                background-color: transparent;
                border: none;
            }}
        """)

        # Increased padding for more breathing room
        padding_v = 10
        padding_h = 15
        if self.is_user:
            layout.setContentsMargins(padding_h, padding_v, padding_h + self.tail_width, padding_v)
        else:
            layout.setContentsMargins(padding_h + self.tail_width, padding_v, padding_h, padding_v)

        layout.addWidget(self.label)
        self.setLayout(layout)
        self.setMaximumWidth(380) # Increased max width
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bubble_rect = self.rect().adjusted(
            self.tail_width if not self.is_user else 0, 0,
            -self.tail_width if self.is_user else 0, 0
        )
        path = QPainterPath()
        path.addRoundedRect(bubble_rect, self.bubble_radius, self.bubble_radius)
        if self.is_user:
            p1 = QPoint(bubble_rect.right() - self.bubble_radius, bubble_rect.bottom())
            p2 = QPoint(bubble_rect.right() + self.tail_width, bubble_rect.bottom())
            p3 = QPoint(bubble_rect.right(), bubble_rect.bottom() - self.tail_height)
            path.addPolygon(QPolygon([p1, p2, p3]))
        else:
            p1 = QPoint(bubble_rect.left() + self.bubble_radius, bubble_rect.bottom())
            p2 = QPoint(bubble_rect.left() - self.tail_width, bubble_rect.bottom())
            p3 = QPoint(bubble_rect.left(), bubble_rect.bottom() - self.tail_height)
            path.addPolygon(QPolygon([p1, p2, p3]))
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.user_color if self.is_user else self.ai_color)
        painter.drawPath(path)

class ChatWindow(QWidget):
    add_message_signal = Signal(str, bool)
    audio_level_updated_signal = Signal(int)

    def __init__(self, tray_icon, voice_threshold):
        super().__init__()
        self.tray_icon = tray_icon
        self.voice_threshold = voice_threshold
        self.initUI()
        self.add_message_signal.connect(self.add_message)
        self.audio_level_updated_signal.connect(self.update_voice_indicator)
        self.queue = Queue()
        self.session = None

    def initUI(self):
        self.setWindowTitle('YourOwnWaifu')
        self.setGeometry(100, 100, 350, 700) # Increased width slightly
        self.setStyleSheet(f"background-color: {BACKGROUND_SECONDARY_COLOR};")
        
        QApplication.setFont(QFont("Segoe UI", 10))

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_bar = QWidget()
        top_bar.setStyleSheet(f"background-color: {TOP_BAR_CHAT_COLOR}; border-bottom: 1px solid {TOP_BAR_CHAT_BORDER_COLOR};")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 10, 12, 10)
        self.title_label = QLabel("Loading model, please wait...")
        self.title_label.setStyleSheet(f"color: {TEXT_PRIMARY_COLOR}; font-size: 15px; font-weight: 500;")
        self.title_label.setToolTip("Click the ❌ (close) button to minimize to tray and run in background")

        self.voice_indicator = VoiceIndicatorWidget(threshold=self.voice_threshold)
        self.voice_indicator.setToolTip("Voice activity indicator - shows microphone input level")
        voice_indicator_label = QLabel("Mic:")
        voice_indicator_label.setStyleSheet(f"color: {TEXT_PRIMARY_COLOR}; font-size: 13px; margin-right: 2px;")
        
        # Add mute button
        self.mute_button = QPushButton("🔊")
        self.is_muted = False
        self.mute_button.setFixedSize(32, 24)
        self.mute_button.setStyleSheet(f"""
            QPushButton {{
                border: 1px solid {SEPARATOR_COLOR}; border-radius: 7px;
                background-color: {BACKGROUND_SECONDARY_COLOR}; color: {TEXT_PRIMARY_COLOR};
                font-size: 14px; font-weight: normal;
            }}
            QPushButton:hover {{ background-color: #F0F0F5; border-color: {PRIMARY_COLOR}; }}
            QPushButton:pressed {{ background-color: #E0E0E5; }}
        """)
        self.mute_button.setToolTip("Toggle microphone on/off")
        self.mute_button.clicked.connect(self.toggle_mute)

        control_button_style = f"""
            QPushButton {{
                min-width: 24px; min-height: 24px; font-size: 15px; font-weight: normal;
                border: 1px solid {SEPARATOR_COLOR}; border-radius: 7px;
                background-color: {BACKGROUND_SECONDARY_COLOR}; color: {TEXT_PRIMARY_COLOR};
            }}
            QPushButton:hover {{ background-color: #F0F0F5; border-color: {PRIMARY_COLOR}; }}
            QPushButton:pressed {{ background-color: #E0E0E5; }}
        """
        control_label_style = f"color: {TEXT_PRIMARY_COLOR}; font-size: 13px; margin-right: 5px;"
        volume_layout = QHBoxLayout(); volume_buttons_widget = QWidget()
        volume_buttons = QVBoxLayout(volume_buttons_widget); volume_buttons.setContentsMargins(0,0,0,0); volume_buttons.setSpacing(3)
        volume_plus = QPushButton("+"); volume_minus = QPushButton("-")
        for btn in [volume_plus, volume_minus]: btn.setStyleSheet(control_button_style)
        volume_plus.setToolTip("Increase waifu voice volume")
        volume_minus.setToolTip("Decrease waifu voice volume")
        volume_plus.clicked.connect(self.increase_volume); volume_minus.clicked.connect(self.decrease_volume)
        volume_buttons.addWidget(volume_plus); volume_buttons.addWidget(volume_minus)
        volume_label = QLabel("Volume:"); volume_label.setStyleSheet(control_label_style)
        volume_layout.addWidget(volume_label); volume_layout.addWidget(volume_buttons_widget); volume_layout.setSpacing(5)
        size_layout = QHBoxLayout(); size_buttons_widget = QWidget()
        size_buttons = QVBoxLayout(size_buttons_widget); size_buttons.setContentsMargins(0,0,0,0); size_buttons.setSpacing(3)
        size_plus = QPushButton("+"); size_minus = QPushButton("-")
        for btn in [size_plus, size_minus]: btn.setStyleSheet(control_button_style)
        size_plus.setToolTip("Increase waifu model size")
        size_minus.setToolTip("Decrease waifu model size")
        size_plus.clicked.connect(self.increase_size); size_minus.clicked.connect(self.decrease_size)
        size_buttons.addWidget(size_plus); size_buttons.addWidget(size_minus)
        size_label = QLabel("Size:"); size_label.setStyleSheet(control_label_style)
        size_layout.addWidget(size_label); size_layout.addWidget(size_buttons_widget); size_layout.setSpacing(5)
        controls_layout = QHBoxLayout(); controls_layout.addLayout(volume_layout)
        controls_layout.addSpacing(15); controls_layout.addLayout(size_layout)
        
        # Add help button for features info
        self.help_chat_button = QPushButton("?")
        self.help_chat_button.setFixedSize(22, 22)
        self.help_chat_button.setStyleSheet(f"""
            QPushButton {{ font-size: 14px; font-weight: bold; background-color: #E0E0E5;
                           color: {TEXT_PRIMARY_COLOR}; border: 1px solid {SEPARATOR_COLOR}; border-radius: 11px; }}
            QPushButton:hover {{ background-color: {SEPARATOR_COLOR}; border-color: {PRIMARY_COLOR}; }} """)
        self.help_chat_button.setCursor(Qt.PointingHandCursor)
        self.help_chat_button.setToolTip("Click for app features & tips")
        self.help_chat_button.clicked.connect(self.show_features_info)
        
        self.exit_button = QPushButton("EXIT")
        # Restyled exit button to be less jarring
        self.exit_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #FEEBEA;
                color: {ERROR_COLOR};
                border: 1px solid #FDDAD7;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {ERROR_COLOR};
                color: white;
                border-color: {ERROR_COLOR};
            }}
            QPushButton:pressed {{
                background-color: #D32F2F;
                border-color: #D32F2F;
            }}
        """)
        self.exit_button.setToolTip("Exit the application completely")
        self.exit_button.clicked.connect(self.terminate)
        top_layout.addWidget(self.title_label); top_layout.addStretch(1)
        top_layout.addWidget(voice_indicator_label)
        top_layout.addWidget(self.voice_indicator)
        top_layout.addSpacing(5)
        top_layout.addWidget(self.mute_button)
        top_layout.addSpacing(15)
        top_layout.addLayout(controls_layout)
        top_layout.addSpacing(10)
        top_layout.addWidget(self.help_chat_button)
        top_layout.addSpacing(15)
        top_layout.addWidget(self.exit_button)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{ border: none; background-color: transparent; }}
            QScrollBar:vertical {{ border: none; background: {BACKGROUND_PRIMARY_COLOR}; width: 8px; margin: 0; }}
            QScrollBar::handle:vertical {{ background: {SEPARATOR_COLOR}; min-height: 20px; border-radius: 4px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
        """)
        self.chat_widget = QWidget()
        self.chat_widget.setObjectName("chatWidgetArea")
        chat_bg_path = "files/background2.png"
        if os.path.exists(chat_bg_path):
            abs_chat_bg_path = os.path.abspath(chat_bg_path).replace("\\", "/")
            self.chat_widget.setStyleSheet(f"QWidget#chatWidgetArea {{ background-image: url('{abs_chat_bg_path}'); background-repeat: repeat-xy; background-position: center; }}")
        else:
            self.chat_widget.setStyleSheet(f"QWidget#chatWidgetArea {{ background-color: {BACKGROUND_SECONDARY_COLOR}; }}")
        
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.setSpacing(10) # Increased spacing for readability
        self.chat_layout.setContentsMargins(15, 15, 15, 15)
        self.chat_layout.addStretch() # This pushes all content to the top
        self.scroll_area.setWidget(self.chat_widget)

        input_widget = QWidget()
        input_widget.setStyleSheet(f"background-color: {BACKGROUND_PRIMARY_COLOR}; border-top: 1px solid {SEPARATOR_COLOR};")
        input_layout = QHBoxLayout(input_widget)
        input_layout.setContentsMargins(12, 8, 12, 8); input_layout.setSpacing(12)
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type a message...")
        self.message_input.setToolTip("Type your message here or use voice input (if enabled)")
        self.message_input.setStyleSheet(f"""
            QLineEdit {{
                border: 1px solid {SEPARATOR_COLOR}; border-radius: 18px; padding: 10px 15px;
                font-size: 15px; background-color: {BACKGROUND_SECONDARY_COLOR}; color: {TEXT_PRIMARY_COLOR};
            }}
            QLineEdit:focus {{ border: 1.5px solid {PRIMARY_COLOR}; }}
        """)
        self.send_button = QPushButton("➤")
        self.send_button.setFixedSize(38, 38)
        self.send_button.setToolTip("Send message")
        self.send_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_COLOR}; color: white; border: none;
                border-radius: 19px; font: bold 20px; padding-bottom: 2px;
            }}
            QPushButton:hover {{ background-color: #2DB850; }}
            QPushButton:pressed {{ background-color: #28A448; }}
        """)
        input_layout.addWidget(self.message_input); input_layout.addWidget(self.send_button)
        self.send_button.clicked.connect(self.send_message)
        self.message_input.returnPressed.connect(self.send_message)

        main_layout.addWidget(top_bar); main_layout.addWidget(self.scroll_area, 1)
        main_layout.addWidget(input_widget)

    def update_voice_indicator(self, level: int):
        if self.voice_indicator:
            self.voice_indicator.set_level(level)

    def add_message(self, message, is_user):
        bubble = ChatMessageWidget(message, is_user)
        wrapper_layout = QHBoxLayout()
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        if is_user:
            wrapper_layout.addStretch()
            wrapper_layout.addWidget(bubble)
        else:
            wrapper_layout.addWidget(bubble)
            wrapper_layout.addStretch()
        
        self.chat_layout.insertLayout(self.chat_layout.count() - 1, wrapper_layout)

        QTimer.singleShot(10, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        """Scrolls the chat area to the maximum vertical position."""
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def set_active_title(self, title_text):
        QTimer.singleShot(0, lambda: self.title_label.setText(title_text))

    def increase_volume(self):
        try:
            if self.session is None:
                sessions = AudioUtilities.GetAllSessions()
                for session in sessions:
                    if session.Process and session.Process.name() == "YourOwnWaifu.exe": self.session = session; break
            if self.session:
                volume = self.session._ctl.QueryInterface(ISimpleAudioVolume)
                new_volume = min(1.0, volume.GetMasterVolume() + 0.1)
                volume.SetMasterVolume(new_volume, None)
        except Exception as e: print(f"Error increasing volume: {e}")

    def decrease_volume(self):
        try:
            if self.session is None:
                sessions = AudioUtilities.GetAllSessions()
                for session in sessions:
                    if session.Process and session.Process.name() == "YourOwnWaifu.exe": self.session = session; break
            if self.session:
                volume = self.session._ctl.QueryInterface(ISimpleAudioVolume)
                new_volume = max(0.0, volume.GetMasterVolume() - 0.1)
                volume.SetMasterVolume(new_volume, None)
        except Exception as e: print(f"Error decreasing volume: {e}")

    def increase_size(self): Live2D.adjustWindowSize(True)
    def decrease_size(self): Live2D.adjustWindowSize(False)

    def add_message_spoken(self, message, user :bool): self.add_message_signal.emit(message, user)
    def send_message(self):
        message = self.message_input.text().strip()
        if message: 
            # Simple debounce mechanism to prevent rapid sending
            from time import time
            current_time = time()
            if hasattr(self, '_last_send_time') and current_time - self._last_send_time < 0.1:
                print("Message send blocked due to debounce")
                return
            self._last_send_time = current_time
            
            # Check if we should interrupt based on audio playback or processing (same logic as audio messages)
            if hasattr(self, 'speech_listener') and self.speech_listener:
                if (self.speech_listener.audio_player.current_playback_active or 
                    self.speech_listener.is_playing.is_set() or 
                    not self.speech_listener.response_queue.empty()):
                    print("Interrupting due to active playback or processing (text message)")
                    self.speech_listener.interrupt_current_response()
                    # Small delay to allow interruption to complete
                    from time import sleep
                    sleep(0.1)  # Increased delay slightly for better synchronization
                
                # Update timing variables (same as audio messages)
                self.speech_listener.last_response = current_time
                self.speech_listener.last_user_talk = current_time
            
            self.add_message_spoken(message, True); self.queue.put(message); self.message_input.clear()
    
    def closeEvent(self, event):
        """Handle window close event properly"""
        print("[GUI] ChatWindow close event triggered")
        if event:
            event.accept()  # Accept the close event instead of ignoring it
        # Trigger app exit through tray icon to ensure proper shutdown
        if hasattr(self, 'tray_icon') and self.tray_icon:
            self.tray_icon.exit_app()
        else:
            # Fallback if tray_icon is not available
            QApplication.quit()
    
    def terminate(self): 
        self.tray_icon.exit_app()

    def toggle_mute(self):
        """Toggle microphone mute state"""
        self.is_muted = not self.is_muted
        
        # Update button appearance
        if self.is_muted:
            self.mute_button.setText("🔇")
            self.mute_button.setStyleSheet(f"""
                QPushButton {{
                    border: 1px solid {ERROR_COLOR}; border-radius: 7px;
                    background-color: #FEEBEA; color: {ERROR_COLOR};
                    font-size: 14px; font-weight: normal;
                }}
                QPushButton:hover {{ background-color: {ERROR_COLOR}; color: white; }}
                QPushButton:pressed {{ background-color: #D32F2F; color: white; }}
            """)
        else:
            self.mute_button.setText("🔊")
            self.mute_button.setStyleSheet(f"""
                QPushButton {{
                    border: 1px solid {SEPARATOR_COLOR}; border-radius: 7px;
                    background-color: {BACKGROUND_SECONDARY_COLOR}; color: {TEXT_PRIMARY_COLOR};
                    font-size: 14px; font-weight: normal;
                }}
                QPushButton:hover {{ background-color: #F0F0F5; border-color: {PRIMARY_COLOR}; }}
                QPushButton:pressed {{ background-color: #E0E0E5; }}
            """)
        
        # Notify the speech listener about mute state change
        if hasattr(self, 'speech_listener') and self.speech_listener:
            self.speech_listener.set_mute_state(self.is_muted)
        
        print(f"Microphone {'muted' if self.is_muted else 'unmuted'}")

    def set_speech_listener(self, speech_listener):
        """Set the speech listener reference for mute control and interruption handling"""
        self.speech_listener = speech_listener

    def show_features_info(self):
        """Show comprehensive information about app features and capabilities"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("YourOwnWaifu Features & Tips")
        msg_box.setIcon(QMessageBox.Information)
        info_text = (
            "<b>🗣️ Your AI Waifu's Capabilities:</b><br><br>"
            "• <b>Web Search</b> - Finds current information online<br>"
            "• <b>Image Generation</b> - Creates pictures based on your requests<br>"
            "• <b>Screen Analysis</b> - Can see and analyze your screen<br>"
            "• <b>Website Opening</b> - Opens URLs in your browser<br>"
            "• <b>Python Code Execution</b> - Runs code to help with tasks<br><br>"
            "• <b>Answer To Clipbord</b> - Gives you an answer straight to copy memory for you to paste<br><br>"
            "<i>To use these features, just ask nicely!</i>"
        )
        msg_box.setText(info_text)
        
        detailed_text = (
            "<b>💬 Usage Tips:</b><br>"
            "• Click ❌ to minimize to chat window (keeps running in background)<br>"
            "• Use EXIT button to completely close the application<br>"
            "• Your conversations are processed securely and privately<br><br>"
            "<b>⚠️ Demo Mode Notice:</b><br>"
            "If features aren't working properly, switch to your own Google AI API key for full functionality.<br><br>"
            "<b>⚖️ License & Credits:</b><br>"
            "Version: 0.1<br>"
            "This application is provided for free. Redistribution is permitted, but charging money for it is strictly prohibited.<br>"
            "Official Website: <a href='https://yourownwaifu.com'>yourownwaifu.com</a>"
        )
        msg_box.setInformativeText(detailed_text)
        msg_box.setTextFormat(Qt.RichText)
        msg_box.findChild(QLabel, "qt_msgbox_label").setOpenExternalLinks(True)
        msg_box.findChild(QLabel, "qt_msgbox_informativelabel").setOpenExternalLinks(True)
        
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.setStyleSheet(f"""
            QMessageBox {{ background-color: {BACKGROUND_PRIMARY_COLOR}; }}
            QLabel {{ color: {TEXT_PRIMARY_COLOR}; font-size: 14px; }}
            QPushButton {{
                background-color: {PRIMARY_COLOR}; color: white; border: none;
                border-radius: 5px; padding: 8px 16px; font-size: 14px; min-width: 80px;
            }}
            QPushButton:hover {{ background-color: #0062CC; }}
        """)
        msg_box.exec()

    

class FancyApp(QWidget):
    def __init__(self, tray_icon, api, is_available):
        super().__init__()
        guiis_available = is_available; self.api_key = api; self.return_data = None
        self.started: bool = False; self.models = load_characters(); self.tray_icon = tray_icon
        self.chat_window = None; self.initUI(); self.stop_event = Event()
        self.unique_id = "Live2D"
        UserData.unique_id = self.unique_id
        self.model = "gemini-2.5-flash"; self.fast_model = "gemini-2.5-flash-lite-preview-06-17"
        self.usage_reported = 0

    def initUI(self):
        self.setWindowTitle('YourOwnWaifu - Setup'); self.setWindowIcon(QIcon("files/icon.ico"))
        self.setGeometry(0, 0, 450, 650) 
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        self.move((screen_geometry.width() - self.width()) // 2, (screen_geometry.height() - self.height()) // 2)

        self.background_label = QLabel(self)
        bg_pixmap_path = "files/background.png"
        if os.path.exists(bg_pixmap_path):
            bg_pixmap = QPixmap(bg_pixmap_path)
            if not bg_pixmap.isNull():
                self.background_label.setPixmap(bg_pixmap)
                self.background_label.setScaledContents(True)
            else: self.background_label.setStyleSheet(f"background-color: {BACKGROUND_PRIMARY_COLOR};")
        else: self.background_label.setStyleSheet(f"background-color: {BACKGROUND_PRIMARY_COLOR};")
        self.background_label.setGeometry(self.rect())

        main_scroll_area = QScrollArea(self)
        main_scroll_area.setWidgetResizable(True)
        main_scroll_area.setStyleSheet("border: none; background: transparent;")

        container_widget = QWidget()
        container_widget.setStyleSheet("background: transparent;")
        main_layout = QVBoxLayout(container_widget)
        main_layout.setContentsMargins(25, 20, 25, 20); main_layout.setSpacing(15)
        main_layout.setAlignment(Qt.AlignTop)

        logo_container = QWidget(); logo_layout = QHBoxLayout(logo_container)
        logo_layout.setAlignment(Qt.AlignCenter); logo = QLabel()
        icon_path = "files/icon.png"
        if os.path.exists(icon_path):
            logo_pixmap = QPixmap(icon_path)
            if not logo_pixmap.isNull(): logo.setPixmap(logo_pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else: logo.setText("YourOwnWaifu"); logo.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {PRIMARY_COLOR};")
        else: logo.setText("YourOwnWaifu"); logo.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {PRIMARY_COLOR};")
        logo_layout.addWidget(logo); main_layout.addWidget(logo_container)
        
        def create_input_group(placeholder, default_text=""):
            input_field = QLineEdit(default_text); input_field.setPlaceholderText(placeholder); input_field.setFixedHeight(40)
            input_field.setStyleSheet(f"""
                QLineEdit {{ background-color: {BACKGROUND_SECONDARY_COLOR}; border: 1px solid {SEPARATOR_COLOR};
                            border-radius: 8px; color: {TEXT_PRIMARY_COLOR}; padding: 0 12px; font-size: 15px; }}
                QLineEdit:focus {{ border: 1.5px solid {PRIMARY_COLOR}; }}""")
            return input_field
        self.username_input = create_input_group("Enter your Username")
        self.api_key_input = create_input_group("Enter your Gemini API Key"); self.api_key_input.setEchoMode(QLineEdit.Password)

        def create_dropdown_group(label_text, options_dict_or_list):
            label = QLabel(label_text); label.setStyleSheet(f"color: {TEXT_PRIMARY_COLOR}; font-size: 15px; font-weight: 500; margin-bottom: 5px;")
            dropdown = QComboBox(); dropdown.setFixedHeight(40)
            if isinstance(options_dict_or_list, dict): dropdown.addItems(options_dict_or_list.keys())
            else: dropdown.addItems(options_dict_or_list)
            dropdown.setStyleSheet(f"""
                QComboBox {{ background-color: {BACKGROUND_SECONDARY_COLOR}; border: 1px solid {SEPARATOR_COLOR};
                            border-radius: 8px; color: {TEXT_PRIMARY_COLOR}; padding: 0 12px; font-size: 15px; }}
                QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right; width: 30px;
                                        border-left-width: 1px; border-left-color: {SEPARATOR_COLOR}; border-left-style: solid;
                                        border-top-right-radius: 8px; border-bottom-right-radius: 8px; }}
                QComboBox QAbstractItemView {{ background-color: {BACKGROUND_SECONDARY_COLOR}; border: 1px solid {SEPARATOR_COLOR};
                                                color: {TEXT_PRIMARY_COLOR}; selection-background-color: {PRIMARY_COLOR};
                                                selection-color: white; padding: 5px; border-radius: 4px; }} """)
            group_layout = QVBoxLayout(); group_layout.setSpacing(3); group_layout.addWidget(label); group_layout.addWidget(dropdown)
            return group_layout, dropdown

        choices = list(self.models.keys())
        if not choices: choices = ["No models found! - check Resources folder"]
        
        model_layout, self.model_dropdown = create_dropdown_group("Waifu Model:", choices)
        
        # Add credits button next to model dropdown
        model_with_credits_layout = QHBoxLayout()
        model_with_credits_layout.setSpacing(8)
        model_with_credits_layout.addLayout(model_layout, 1)
        
        self.credits_button = QPushButton("?")
        self.credits_button.setFixedSize(22, 22)
        self.credits_button.setToolTip("View model credits and attribution")
        self.credits_button.setStyleSheet(f"""
            QPushButton {{ font-size: 14px; font-weight: bold; background-color: #E0E0E5;
                           color: {TEXT_PRIMARY_COLOR}; border: 1px solid {SEPARATOR_COLOR}; border-radius: 11px; }}
            QPushButton:hover {{ background-color: {SEPARATOR_COLOR}; border-color: {PRIMARY_COLOR}; }} """)
        self.credits_button.setCursor(Qt.PointingHandCursor)
        self.credits_button.clicked.connect(self.show_model_credits)
        model_with_credits_layout.addWidget(self.credits_button, 0, Qt.AlignBottom)
        
        language_layout, self.language_dropdown = create_dropdown_group("Speaking Language:", AVAILABLE_LANGUAGES)

        sensitivity_label = QLabel("Voice Detection Sensitivity")
        sensitivity_label.setStyleSheet(f"color: {TEXT_PRIMARY_COLOR}; font-size: 15px; font-weight: 500; margin-bottom: 5px;")
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(0, 100)
        self.sensitivity_slider.setValue(50) 
        self.sensitivity_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb; background: white; height: 10px; border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: #007AFF; border: 1px solid #007AFF; height: 10px; border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #eee, stop:1 #ccc);
                border: 1px solid #777; width: 18px; margin-top: -4px; margin-bottom: -4px; border-radius: 9px;
            } """)
        sensitivity_value_layout = QHBoxLayout()
        sensitivity_low_label = QLabel("Less Sensitive"); sensitivity_high_label = QLabel("More Sensitive")
        sensitivity_low_label.setAlignment(Qt.AlignLeft); sensitivity_high_label.setAlignment(Qt.AlignRight)
        sensitivity_value_layout.addWidget(sensitivity_low_label); sensitivity_value_layout.addWidget(sensitivity_high_label)
        sensitivity_layout = QVBoxLayout(); sensitivity_layout.setSpacing(3)
        sensitivity_layout.addWidget(sensitivity_label); sensitivity_layout.addWidget(self.sensitivity_slider)
        sensitivity_layout.addLayout(sensitivity_value_layout)

        checkbox_style = f"""
            QCheckBox {{ color: {TEXT_PRIMARY_COLOR}; font-size: 14px; background-color: transparent; spacing: 8px; }}
            QCheckBox::indicator {{ width: 20px; height: 20px; border: 1.5px solid {SEPARATOR_COLOR};
                                    border-radius: 5px; background-color: {BACKGROUND_SECONDARY_COLOR}; }}
            QCheckBox::indicator:hover {{ border-color: {PRIMARY_COLOR}; }}
            QCheckBox::indicator:checked {{ background-color: {PRIMARY_COLOR}; border-color: {PRIMARY_COLOR}; }}
            QCheckBox::indicator:checked:hover {{ background-color: #0062CC; border-color: #0062CC; }} """
        self.Live2D_checkbox = QCheckBox("Use Demo Mode (Limited Usage)")
        self.Live2D_checkbox.setStyleSheet(checkbox_style); self.Live2D_checkbox.toggled.connect(self.toggle_Live2D_mode)
        Live2D_layout = QHBoxLayout(); Live2D_layout.setSpacing(8); Live2D_layout.setContentsMargins(0, 0, 0, 0)
        Live2D_layout.addWidget(self.Live2D_checkbox)
        self.help_button = QPushButton("?"); self.help_button.setFixedSize(22, 22)
        self.help_button.setStyleSheet(f"""
            QPushButton {{ font-size: 14px; font-weight: bold; background-color: #E0E0E5;
                           color: {TEXT_PRIMARY_COLOR}; border: 1px solid {SEPARATOR_COLOR}; border-radius: 11px; }}
            QPushButton:hover {{ background-color: {SEPARATOR_COLOR}; border-color: {PRIMARY_COLOR}; }} """)
        self.help_button.setCursor(Qt.PointingHandCursor)
        self.help_button.clicked.connect(self.show_setup_help)
        Live2D_layout.addWidget(self.help_button); Live2D_layout.addStretch(1)
        self.checkbox = QCheckBox("Turn microphone off (Text input only)")
        self.checkbox.setStyleSheet(checkbox_style)
        self.start_button = QPushButton("START WAIFU"); self.start_button.setFixedHeight(42)

        self.start_button.setStyleSheet(f"""
            QPushButton {{ background-color: {ACCENT_COLOR}; color: white; border: none; border-radius: 8px;
                            padding: 0 20px; font-size: 16px; font-weight: bold; }}
            QPushButton:hover {{ background-color: #2DB850; }} QPushButton:pressed {{ background-color: #28A448; }}""")
        self.start_button.clicked.connect(self.button_animation); self.start_button.clicked.connect(self.start_app)

        main_layout.addWidget(self.username_input); main_layout.addWidget(self.api_key_input)
        main_layout.addLayout(Live2D_layout)
        
        dropdowns_grid = QHBoxLayout()
        dropdowns_grid.addLayout(model_with_credits_layout, 1)
        dropdowns_grid.addLayout(language_layout, 1)
        main_layout.addLayout(dropdowns_grid)
        
        main_layout.addLayout(sensitivity_layout) 
        main_layout.addWidget(self.checkbox)
        main_layout.addSpacing(15)
        main_layout.addWidget(self.start_button)
        main_layout.addStretch(1)
        main_scroll_area.setWidget(container_widget)
        full_window_layout = QVBoxLayout(self); full_window_layout.setContentsMargins(0,0,0,0)
        full_window_layout.addWidget(main_scroll_area); self.setLayout(full_window_layout)
        self.load_settings(); self.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.background_label.setGeometry(self.rect())
    def show_model_credits(self):
            """Show credits for the models and Live2D"""
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Model Credits")
            msg_box.setIcon(QMessageBox.Information)
            
            main_text = (
                "<b>🎭 Model Credits & Attribution</b><br>"
                "This application uses beautiful Live2D models created by talented artists:"
            )

            detailed_text = (
                "<b>📚 Model Sources:</b><br>"
                "• <b>GothL2D<b>:"
                "  <a href='https://sadwhale-studios.itch.io/goth-mofu'>sadwhale-studios.itch.io/goth-mofu</a><br><br>"

                "• <b>Frieren:"
                "  <a href='https://ko-fi.com/s/2deea6c774'>ko-fi.com/s/2deea6c774</a><br><br>"

                "• <b>Additional Models:</b> Other models in the Resources folder are powered from  <a href='www.live2d.com'>Live2D</a><br><br>"

                "<b>➕ Add Your Own Models:</b><br>"
                "You can add your own Live2D models to the application!<br>"
                "Simply place your model's folder into the 'Resources' folder, which is located in the same directory as the application.<br>"
                "Once added, you should be able to select and interact with your model.<br><br>"
                "<b>Important:</b><br>"
                "• Ensure your model's folder structure is similar to the existing model folders within 'Resources'.<br>"
                "• The name of your model's folder **must** be the same as the main model file (e.g., if your model file is `Frieren.moc3`, the folder should be named `Frieren`).<br><br>"

                "<b>🔧 Technology:</b><br>"
                "• <b>Live2D:</b> This application uses Live2D technology for model rendering and animation<br>"
                "  <a href='https://www.live2d.com/'>live2d.com</a><br><br>"
            )

            msg_box.setText(main_text)
            msg_box.setInformativeText(detailed_text)
            msg_box.setTextFormat(Qt.RichText)
            msg_box.findChild(QLabel, "qt_msgbox_label").setOpenExternalLinks(True)
            msg_box.findChild(QLabel, "qt_msgbox_informativelabel").setOpenExternalLinks(True)
            
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setStyleSheet(f"""
                QMessageBox {{ background-color: {BACKGROUND_PRIMARY_COLOR}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY_COLOR}; }}
                QMessageBox QPushButton {{ 
                    background-color: {PRIMARY_COLOR}; color: white; border: none; 
                    border-radius: 6px; padding: 8px 16px; font-weight: bold; 
                }}
                QMessageBox QPushButton:hover {{ background-color: #0062CC; }}
            """)
            
            msg_box.exec()
    # MODIFIED: Renamed from show_api_key_info and content expanded
    def show_setup_help(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("About YourOwnWaifu")
        msg_box.setIcon(QMessageBox.Information)
        
        main_text = (
            "<b>🚀 Demo Mode vs. API Key</b><br>"
            "Using your own API key is fhighly recommended. The <b>Demo Mode</b> uses a shared proxy which may be slow or unavailable due to high traffic. "
            "Demo Mode remains available, though with a limited number of messages. "
            "Getting your own Google AI API key is <b>free, fast (~2 minutes), and removes all limitations.</b><br>"
            f"Click here to get your key: <a href='https://aistudio.google.com/apikey'>aistudio.google.com/apikey</a>"
        )
        
        detailed_text = (
            "<hr>"
            "<b>⚖️ License & Credits</b><br>"
            "Version: 0.1<br>"
            "This application is provided for free. You may redistribute it, but charging money for it is strictly prohibited.<br>"
            "Official Website: <a href='https://yourownwaifu.com'>yourownwaifu.com</a><br><br>"
            
            "<b>🔒 Data Safety & Privacy</b><br>"
            "Your privacy is important. Here's how your data is handled:<br>"
            "• <b>Settings & API Key:</b> All settings, including your API key, are saved locally on your computer in a `setting.json` file. They are never sent anywhere else.<br>"
            "• <b>Conversations:</b> If you use your own API key, your chats are sent directly to Google's AI services. If you use Demo Mode, chats pass through our proxy server (it has to be like that) but are not stored or logged."
        )

        msg_box.setText(main_text)
        msg_box.setInformativeText(detailed_text)
        msg_box.setTextFormat(Qt.RichText)
        msg_box.findChild(QLabel, "qt_msgbox_label").setOpenExternalLinks(True)
        msg_box.findChild(QLabel, "qt_msgbox_informativelabel").setOpenExternalLinks(True)
        
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.setStyleSheet(f"""
            QMessageBox {{ background-color: {BACKGROUND_PRIMARY_COLOR}; }}
            QLabel {{ color: {TEXT_PRIMARY_COLOR}; font-size: 14px; }}
            QPushButton {{
                background-color: {PRIMARY_COLOR}; color: white; border: none;
                border-radius: 5px; padding: 8px 16px; font-size: 14px; min-width: 80px;
            }}
            QPushButton:hover {{ background-color: #0062CC; }}
        """)
        msg_box.exec()

    def toggle_Live2D_mode(self, checked):
        self.api_key_input.setEnabled(not checked)
        if checked:
            self.api_key_input.setEchoMode(QLineEdit.Normal)
            self.api_key_input.setText("Demo Mode Active - No API Key Required")
            UserData.is_Live2D = True
        else:
            self.api_key_input.setEchoMode(QLineEdit.Password)
            self.api_key_input.setText(self.api_key)
            UserData.is_Live2D = False

    def load_settings(self):
        settings = load_settings()
        if settings:
            self.api_key = settings.get("GEMINI_API_KEY", "")
            self.api_key_input.setText(self.api_key)
            self.username_input.setText(settings.get("USERNAME", ""))
            saved_lang_code = settings.get("SPEAKING_LANGUAGE", "en")
            for name, code in AVAILABLE_LANGUAGES.items():
                if code == saved_lang_code: self.language_dropdown.setCurrentText(name); break
            
            model_name = settings.get("MODEL_NAME", "")
            if model_name:
                for i in range(self.model_dropdown.count()):
                    if model_name in self.model_dropdown.itemText(i): self.model_dropdown.setCurrentIndex(i); break
            self.checkbox.setChecked(settings.get("MICROPHONE_OFF", False))
            self.Live2D_checkbox.setChecked(settings.get("Live2D_MODE", False))
            self.sensitivity_slider.setValue(settings.get("VOICE_SENSITIVITY", 50))

    def button_animation(self):
        animation = QPropertyAnimation(self.start_button, b"geometry"); animation.setDuration(100)
        animation.setEasingCurve(QEasingCurve.OutQuad); start_rect = self.start_button.geometry()
        end_rect = start_rect.adjusted(-2, -1, 2, 1)
        animation.setKeyValueAt(0, start_rect); animation.setKeyValueAt(0.5, end_rect); animation.setKeyValueAt(1, start_rect)
        animation.start(QPropertyAnimation.DeleteWhenStopped)

    def start_app(self):
        if not self.validate_inputs():
            return

        if self.Live2D_checkbox.isChecked():
            QMessageBox.critical(self, "Demo not enabled", "Demo is deleted in the open versions for security reasons.")
            return
                    
        self.api_key = self.api_key_input.text()
        if not self.api_key.strip():
            QMessageBox.warning(self, "Input Error", "API Key cannot be empty.")
            return
        self.open_chat_window()

    def open_chat_window(self):
        try:
            self.username = self.username_input.text()
            selected_lang_code = AVAILABLE_LANGUAGES.get(self.language_dropdown.currentText(), "en")

            slider_val = self.sensitivity_slider.value()
            min_thresh, max_thresh = 8000, 500
            voice_threshold = int(min_thresh - (slider_val / 100.0) * (min_thresh - max_thresh))

            self.chat_window = ChatWindow(self.tray_icon, voice_threshold) 
            self.chat_window.set_active_title(f"YourOwnWaifu")
            self.tray_icon.show(); self.chat_window.show(); self.started = True

            model_name = self.model_dropdown.currentText().split(" (")[0]
            if model_name not in self.models:
                if self.models: model_name = list(self.models.keys())[0]
                else: QMessageBox.critical(self, "Error", "No character models found."); return

            self.return_data = (2, self.checkbox.isChecked(),
                                self.api_key, self.username, model_name,
                                self.models[model_name], self.model, self.fast_model, selected_lang_code,
                                self.Live2D_checkbox.isChecked(), self.usage_reported,
                                voice_threshold) 

            self.tray_icon.start_event.set()
            
            settings_to_save = {
                "GEMINI_API_KEY": "" if self.Live2D_checkbox.isChecked() else self.api_key,
                "USERNAME": self.username,
                "SPEAKING_LANGUAGE": selected_lang_code,
                "MODEL_NAME": model_name,
                "MICROPHONE_OFF": self.checkbox.isChecked(),
                "Live2D_MODE": self.Live2D_checkbox.isChecked(),
                "VOICE_SENSITIVITY": self.sensitivity_slider.value()
            }
            save_settings(settings_to_save)
            self.hide()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error on startup: {e}")

    def validate_inputs(self):
        if not self.models: QMessageBox.warning(self, "Model Error", "No character models loaded."); return False
        return True


    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit', "Exit YourOwnWaifu?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes: self.tray_icon.exit_app(); event.accept()
        else: event.ignore()

class SystemTrayIcon(QSystemTrayIcon):
    def __init__(self, icon, is_available, parent=None):
        super().__init__(icon, parent)
        self.start_event = Event(); self.stop_event = Event(); self.setToolTip('YourOwnWaifu')
        menu = QMenu(parent)
        self.open_chat_action = menu.addAction("Open Chat"); self.open_chat_action.triggered.connect(self.open_chat_window_from_tray); self.open_chat_action.setEnabled(False)
        self.open_settings_action = menu.addAction("Settings"); self.open_settings_action.triggered.connect(self.open_settings_window_from_tray)
        menu.addSeparator(); exit_action = menu.addAction("Exit"); exit_action.triggered.connect(self.exit_app)
        self.setContextMenu(menu); api_key_value = ""
        try:
            settings_path = os.path.join(getcwd(), 'files', 'setting.json')
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f: api_key_value = json.load(f).get('GEMINI_API_KEY', "")
        except Exception: pass
        self.activated.connect(self.onTrayIconActivated)
        self.main_window = FancyApp(self, api_key_value, is_available); self.main_window.show()

    def onTrayIconActivated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.main_window and self.main_window.chat_window and self.main_window.chat_window.isVisible(): self.main_window.chat_window.activateWindow()
            elif self.main_window and self.main_window.chat_window: self.main_window.chat_window.show(); self.main_window.chat_window.activateWindow()
            elif self.main_window and self.main_window.isVisible(): self.main_window.activateWindow()
            else: self.open_settings_window_from_tray()

    def open_chat_window_from_tray(self):
        if self.main_window and self.main_window.chat_window:
            self.main_window.chat_window.show(); self.main_window.chat_window.activateWindow()

    def open_settings_window_from_tray(self):
        if self.main_window: self.main_window.show(); self.main_window.activateWindow()

    def exit_app(self):
        """Improved exit method with better error handling"""
        print("[GUI] Initiating app exit...")
        try:
            self.stop_event.set()
            
            # Close main window gracefully
            if self.main_window:
                try: 
                    self.main_window.closeEvent = lambda event: event.accept()
                    self.main_window.close()
                except Exception as e:
                    print(f"Error closing main window: {e}")
            
            # Close chat window if it exists
            if hasattr(self, 'main_window') and self.main_window and hasattr(self.main_window, 'chat_window'):
                try:
                    if self.main_window.chat_window:
                        self.main_window.chat_window.close()
                except Exception as e:
                    print(f"Error closing chat window: {e}")
            
            # Hide system tray icon
            try:
                self.hide()
            except Exception as e:
                print(f"Error hiding tray icon: {e}")
            
            # Quit application
            try:
                QApplication.quit()
            except Exception as e:
                print(f"Error quitting QApplication: {e}")
                
        except Exception as e:
            print(f"Error in exit_app: {e}")
            # Force quit as last resort
            try:
                QApplication.quit()
            except:
                import sys
                sys.exit(1)

class GUI:
    def __init__(self): 
        self.app = None
        self.gui = None
        self._exit_called = False
        self._app_destroyed = False
        
    def run_GUI(self, is_available):
        try:
            self.app = QApplication.instance() or QApplication(sys.argv)
            icon_path = "files/icon.ico"
            if os.path.exists(icon_path): 
                self.app.setWindowIcon(QIcon(icon_path))
            self.app.setQuitOnLastWindowClosed(False)
            QApplication.setFont(QFont("Segoe UI", 9))
            
            # Style tooltips to match app theme instead of default black appearance
            self.app.setStyleSheet(f"""
                QToolTip {{
                    background-color: {BACKGROUND_SECONDARY_COLOR};
                    color: {TEXT_PRIMARY_COLOR};
                    border: 1px solid {SEPARATOR_COLOR};
                    border-radius: 6px;
                    padding: 8px;
                    font-size: 11px;
                    font-family: 'Segoe UI';
                }}
            """)
            tray_icon_obj = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
            self.gui = SystemTrayIcon(tray_icon_obj, is_available)
            self.gui.show()
            
            def on_app_started():
                if self.gui and self.gui.main_window and self.gui.main_window.chat_window:
                    self.gui.open_chat_action.setEnabled(True)
                    
            original_wait = self.gui.start_event.wait
            def new_wait_for_start_event(timeout=None):
                res = original_wait(timeout=timeout)
                if res:
                    QTimer.singleShot(0, on_app_started)
                return res
            self.gui.start_event.wait = new_wait_for_start_event
            
            # Install event filter for application-level cleanup
            self.app.aboutToQuit.connect(self._cleanup_before_quit)
            
            # Monitor for application destruction
            def on_app_destroyed():
                self._app_destroyed = True
                print("[GUI] QApplication destroyed")
                
            # Connect to aboutToQuit to set the destruction flag
            self.app.aboutToQuit.connect(on_app_destroyed)
            
            print("[GUI] Starting Qt application event loop")
            exit_code = self.app.exec()
            print(f"[GUI] Qt application event loop exited with code: {exit_code}")
            sys.exit(exit_code)
            
        except Exception as e:
            print(f"Error in run_GUI: {e}")
            self._app_destroyed = True
            sys.exit(1)

    def _cleanup_before_quit(self):
        """Cleanup method called before Qt application quits"""
        print("[GUI] Application about to quit, performing cleanup...")
        self._app_destroyed = True
        try:
            if self.gui:
                self.gui.hide()
        except Exception as e:
            print(f"Error in cleanup: {e}")

    def is_destroyed(self):
        """Check if the GUI application was destroyed"""
        return self._app_destroyed or (self.app is None) or (QApplication.instance() is None)

    def start(self, is_available: bool):
        if not QApplication.instance(): 
            self.run_GUI(is_available)
            
    def exit(self):
        """Improved exit method with safety checks"""
        if self._exit_called:
            return
        self._exit_called = True
        
        print("[GUI] Exit called...")
        try:
            if QApplication.instance() and not self._app_destroyed:
                app = QApplication.instance()
                app.quit()
                # Process events to ensure clean shutdown
                app.processEvents()
        except Exception as e:
            print(f"Error in GUI exit: {e}")
        finally:
            self._app_destroyed = True

interface = GUI()