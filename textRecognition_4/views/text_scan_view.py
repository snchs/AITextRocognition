import flet as ft
import os
from textRecognition_4.model.text_recognition_model import HandwrittenTextRecognizer
from textRecognition_4.utils.utils import create_button, create_card
from Segment_Script import extract_words, extract_lines


class TextScanView:
    def __init__(self, app):
        self.app = app
        model_path = 'model/best_model.pth'

        # создание объекта класса модели
        self.recognizer = HandwrittenTextRecognizer(model_path)



    def show(self, e=None):
        if not self.app.current_user:
            self.app.show_snack_bar("Please login first!")
            return

        self.app.page.controls.clear()

        text_icon = ft.Icon(ft.icons.TEXT_SNIPPET_ROUNDED, size=80, color=ft.colors.BLUE_200)

        header = ft.Text("Handwriting Recognition", style=ft.TextThemeStyle.HEADLINE_LARGE, color=ft.colors.WHITE,
                         weight=ft.FontWeight.BOLD)
        description = ft.Text("Upload an image of handwritten text for recognition.",
                              style=ft.TextThemeStyle.BODY_MEDIUM, color=ft.colors.WHITE70)

        image_input = ft.FilePicker(on_result=self.process_image)
        image_input.file_type_filter = "image/*"
        self.app.page.overlay.append(image_input)

        image_to_seg = ft.FilePicker(on_result=self.segmantate_image)
        image_to_seg.file_type_filter = "image/*"
        self.app.page.overlay.append(image_to_seg)

        Segmentation_button = create_button("Segmantate", ft.icons.SMART_BUTTON, lambda _: image_to_seg.pick_files())

        upload_button = create_button("Upload Image", ft.icons.UPLOAD_FILE, lambda _: image_input.pick_files())

        back_button = ft.IconButton(
            icon=ft.icons.ARROW_BACK,
            icon_color=ft.colors.BLUE_200,
            tooltip="Back to Main Menu",
            on_click=self.app.show_main_menu,
        )

        form = ft.Column(
            [
                text_icon,
                header,
                description,
                ft.Container(height=30),
                upload_button,
                Segmentation_button,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=15,
        )

        card = create_card(ft.Column([
            ft.Container(
                content=back_button,
                alignment=ft.alignment.top_left,
                margin=ft.margin.only(bottom=10)
            ),
            form
        ]))

        self.app.page.add(ft.Container(content=card, alignment=ft.alignment.center, expand=True))
        self.app.page.update()

    def process_image(self, e):
        if not e.files:
            return

        file_path = e.files[0].path
        self.process_and_save_image(file_path)

    def segmantate_image(self, e):
        if not e.files:
            return
        file_path = e.files[0].path
        images = extract_words(file_path)
        lines = extract_lines(file_path)


        text = []
        for image in images:
            text.append(self.recognizer.predict(image))
        self.show_result(text)


    def process_and_save_image(self, file_path):
        file_name = os.path.basename(file_path)

        # сохранение файла в истории анализа пользователя
        image_id = self.app.db_manager.save_image(self.app.current_user[0], file_name)

        predicted_text = self.recognizer.predict(file_path)

        self.app.db_manager.save_analysis_result(image_id, predicted_text)
        # вывод результата
        self.show_result(predicted_text)

    def show_result(self, predicted_text):
        result_content = ft.Column([
            ft.Text("Result", style=ft.TextThemeStyle.HEADLINE_SMALL, color=ft.colors.WHITE,
                    weight=ft.FontWeight.BOLD),
            ft.Text(f"Predicted Text: {predicted_text}", style=ft.TextThemeStyle.BODY_LARGE,
                    color=ft.colors.WHITE),
        ])

        result_card = create_card(result_content)

        new_analysis_button = create_button("New Analysis", ft.icons.ADD, self.show)
        back_to_menu_button = create_button("Back to Menu", ft.icons.HOME, self.app.show_main_menu)

        result_view = ft.Column(
            [
                result_card,
                ft.Container(height=20),
                new_analysis_button,
                back_to_menu_button,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=15,
        )
        self.app.page.controls.clear()
        self.app.page.add(ft.Container(content=result_view, alignment=ft.alignment.center, expand=True))
        self.app.page.update()