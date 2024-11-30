import flet as ft
from database.db_manager import DatabaseManager
from views.login_view import LoginView
from views.register_view import RegisterView
from views.text_scan_view import TextScanView
from views.history_view import HistoryView
from views.drawing_view import DrawingView



class MedicalApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.db_manager = DatabaseManager('scan_text_db.db')
        self.setup_page()
        self.current_user = self.db_manager.check_session(page)

    def setup_page(self):
        self.page.title = "TextScan AI"
        self.page.window.width = 700
        self.page.window.height = 800
        self.page.window.center()
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 50
        self.page.bgcolor = ft.colors.BLUE_GREY_900

    def show_snack_bar(self, message: str):
        self.page.snack_bar = ft.SnackBar(content=ft.Text(message), bgcolor=ft.colors.BLUE_400)
        self.page.snack_bar.open = True
        self.page.update()

    def show_login_view(self, e=None):
        LoginView(self).show()

    def show_register_view(self, e=None):
        RegisterView(self).show()

    def show_textscan_view(self, e=None):
        TextScanView(self).show()

    def analyze_canvas(self, e=None):
        TextScanView(self).process_and_save_image("canvas_image.png")

    def show_history_view(self, e=None):
        HistoryView(self).show()

    def show_drawing_view(self, e=None):
        DrawingView(self).show()

    def logout(self, e=None):
        self.current_user = None
        self.page.client_storage.remove("session_token")
        self.show_snack_bar("You have been logged out.")
        self.show_main_menu()

    def show_main_menu(self, e=None):
        self.page.controls.clear()

        text_icon = ft.Icon(ft.icons.TEXT_SNIPPET_ROUNDED, size=80, color=ft.colors.BLUE_200)

        header = ft.Text("Welcome to TextScan AI", style=ft.TextThemeStyle.HEADLINE_LARGE, color=ft.colors.WHITE,
                         weight=ft.FontWeight.BOLD)
        subheader = ft.Text("Handwriting recognition", style=ft.TextThemeStyle.BODY_MEDIUM,
                            color=ft.colors.WHITE70)
        username_text = ft.Text(f"", style=ft.TextThemeStyle.BODY_MEDIUM,
                                color=ft.colors.WHITE70)
        if self.current_user:
            menu_items = [
                ft.ElevatedButton(
                    content=ft.Row([ft.Icon(ft.icons.DOCUMENT_SCANNER), ft.Text("Handwriting Recognition")],
                                   alignment=ft.MainAxisAlignment.CENTER),
                    on_click=self.show_textscan_view,
                    style=ft.ButtonStyle(color=ft.colors.WHITE, bgcolor=ft.colors.BLUE_400, padding=15,
                                         shape=ft.RoundedRectangleBorder(radius=10)),
                    width=250,
                ),
                ft.ElevatedButton(
                    content=ft.Row([ft.Icon(ft.icons.HISTORY), ft.Text("View History")],
                                   alignment=ft.MainAxisAlignment.CENTER),
                    on_click=self.show_history_view,
                    style=ft.ButtonStyle(color=ft.colors.WHITE, bgcolor=ft.colors.BLUE_400, padding=15,
                                         shape=ft.RoundedRectangleBorder(radius=10)),
                    width=250,
                ),
                ft.ElevatedButton(
                    content=ft.Row([ft.Icon(ft.icons.LOGOUT), ft.Text("Logout")],
                                   alignment=ft.MainAxisAlignment.CENTER),
                    on_click=self.logout,
                    style=ft.ButtonStyle(color=ft.colors.WHITE, bgcolor=ft.colors.RED_400, padding=15,
                                         shape=ft.RoundedRectangleBorder(radius=10)),
                    width=250,
                )
            ]



            username_text = ft.Text(f"username: {self.current_user[1]}", style=ft.TextThemeStyle.BODY_MEDIUM,
                                    color=ft.colors.WHITE70)
        else:
            menu_items = [
                ft.ElevatedButton(
                    content=ft.Row([ft.Icon(ft.icons.LOGIN), ft.Text("Login")], alignment=ft.MainAxisAlignment.CENTER),
                    on_click=self.show_login_view,
                    style=ft.ButtonStyle(color=ft.colors.WHITE, bgcolor=ft.colors.BLUE_400, padding=15,
                                         shape=ft.RoundedRectangleBorder(radius=10)),
                    width=250,
                ),
                ft.ElevatedButton(
                    content=ft.Row([ft.Icon(ft.icons.PERSON_ADD), ft.Text("Register")],
                                   alignment=ft.MainAxisAlignment.CENTER),
                    on_click=self.show_register_view,
                    style=ft.ButtonStyle(color=ft.colors.WHITE, bgcolor=ft.colors.BLUE_400, padding=15,
                                         shape=ft.RoundedRectangleBorder(radius=10)),
                    width=250,
                )
            ]

        menu = ft.Column(
            controls=[
                ft.Container(
                    content=button,
                    alignment=ft.alignment.center,
                ) for button in menu_items
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
        )

        content = ft.Column(
            controls=[text_icon, header, subheader, username_text, ft.Container(height=30), menu],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=15,
        )

        card = ft.Card(
            content=ft.Container(content=content, padding=30),
            elevation=5,
            surface_tint_color=ft.colors.BLUE_GREY_800,
        )

        self.page.add(
            ft.Container(
                content=card,
                alignment=ft.alignment.center,
                expand=True,
            )
        )
        self.page.update()

    def main(self):
        if self.current_user:
            self.show_main_menu()
        else:
            self.show_login_view()


def main(page: ft.Page):
    app = MedicalApp(page)
    app.main()


if __name__ == "__main__":
    ft.app(target=main)
