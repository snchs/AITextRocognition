import flet as ft

class HistoryView:
    def __init__(self, app):
        self.app = app

    def show(self, e=None):
        if not self.app.current_user:
            self.app.show_snack_bar("Please login first!")
            return

        self.app.page.controls.clear()

        text_icon = ft.Icon(ft.icons.HISTORY, size=80, color=ft.colors.BLUE_200)

        header = ft.Text("Analysis History", style=ft.TextThemeStyle.HEADLINE_LARGE, color=ft.colors.WHITE,
                         weight=ft.FontWeight.BOLD)
        subheader = ft.Text("Your previous text detection results", style=ft.TextThemeStyle.BODY_MEDIUM,
                            color=ft.colors.WHITE70)

        history = self.app.db_manager.get_user_history(self.app.current_user[0])

        history_list = ft.ListView(
            spacing=10,
            padding=20,
            controls=[
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text(f"File: {item[0]}", style=ft.TextThemeStyle.BODY_MEDIUM, color=ft.colors.WHITE),
                            ft.Text(f"Result: {item[1]}", style=ft.TextThemeStyle.BODY_MEDIUM, color=ft.colors.WHITE),
                            ft.Text(f"Date: {item[2]}", style=ft.TextThemeStyle.BODY_SMALL, color=ft.colors.WHITE70),
                        ]),
                        padding=15,
                    ),
                    elevation=3,
                    surface_tint_color=ft.colors.BLUE_GREY_700,
                )
                for item in history
            ],
            height=300,
        )

        back_button = ft.IconButton(
            icon=ft.icons.ARROW_BACK,
            icon_color=ft.colors.BLUE_200,
            tooltip="Back to Main Menu",
            on_click=self.app.show_main_menu,
        )

        content = ft.Column(
            [
                text_icon,
                header,
                subheader,
                ft.Container(height=20),
                ft.Container(
                    content=history_list,
                    height=300,
                    border=ft.border.all(1, ft.colors.BLUE_200),
                    border_radius=10,
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=15,
        )

        card = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Container(
                        content=back_button,
                        alignment=ft.alignment.top_left,
                        margin=ft.margin.only(bottom=10)
                    ),
                    content
                ]),
                padding=30
            ),
            elevation=5,
            surface_tint_color=ft.colors.BLUE_GREY_800,
        )

        self.app.page.add(ft.Container(content=card, alignment=ft.alignment.center, expand=True))
        self.app.page.update()