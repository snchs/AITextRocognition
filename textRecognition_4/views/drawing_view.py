import flet as ft
import flet.canvas as cv
from textRecognition_4.utils.utils import create_button
from PIL import Image, ImageDraw


class State:
    x: float
    y: float


class DrawingView:
    def __init__(self, app):
        self.app = app
        self.state = State()
        self.canvas_shapes = []  # Хранение нарисованных линий

    def save_canvas(self, e):
        if not self.canvas_shapes:
            self.app.show_snack_bar("Nothing to save!")
            return

        # Создаем изображение и рисуем линии
        img = Image.new("RGB", (600, 600), "white")
        draw = ImageDraw.Draw(img)

        for shape in self.canvas_shapes:
            if isinstance(shape, cv.Line):
                draw.line(
                    [(shape.x1, shape.y1), (shape.x2, shape.y2)],
                    fill="black",
                    width=3,
                )

        # Сохраняем изображение
        saved_path = "canvas_image.png"
        img.save(saved_path)

    def pan_start(self, e: ft.DragStartEvent):
        self.state.x = e.local_x
        self.state.y = e.local_y

    def pan_update(self, e: ft.DragUpdateEvent):
        line = cv.Line(
            self.state.x,
            self.state.y,
            e.local_x,
            e.local_y,
            paint=ft.Paint(stroke_width=3),
        )
        self.canvas_shapes.append(line)
        self.cp.shapes.append(line)
        self.cp.update()
        self.state.x = e.local_x
        self.state.y = e.local_y

    def show(self, e=None):
        self.app.page.controls.clear()

        # Область рисования
        self.cp = cv.Canvas(
            [
                cv.Fill(
                    ft.Paint(
                        gradient=ft.PaintLinearGradient(
                            (0, 0), (600, 600), colors=[ft.colors.CYAN_50, ft.colors.GREY]
                        )
                    )
                ),
            ],
            content=ft.GestureDetector(
                on_pan_start=self.pan_start,
                on_pan_update=self.pan_update,
                drag_interval=10,
            ),
            expand=False,
        )

        save_button = create_button("Save Drawing", ft.icons.SAVE, self.save_canvas)

        back_button = create_button("Back to TextScanView", ft.icons.ARROW_BACK, self.app.analyze_canvas)

        form = ft.Column(
            [
                ft.Text("Drawing Area", style=ft.TextThemeStyle.HEADLINE_LARGE),
                ft.Text("Draw your input here."),
                ft.Container(height=30),
                ft.Container(
                    self.cp,
                    border_radius=5,
                    width=float("inf"),
                    expand=True,
                ),
                ft.Row([save_button, back_button], alignment=ft.MainAxisAlignment.END),
            ],
            spacing=15,
            expand=True,
        )

        self.app.page.add(ft.Container(content=form, alignment=ft.alignment.center, expand=True))
        self.app.page.update()
