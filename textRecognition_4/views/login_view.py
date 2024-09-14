import flet as ft


class LoginView:
    def __init__(self, app):
        self.app = app

    def show(self):
        self.app.page.controls.clear()
        self.app.page.bgcolor = ft.colors.BLUE_GREY_900
        self.app.page.padding = 50

        text_icon = ft.Icon(ft.icons.TEXT_SNIPPET_ROUNDED, size=80, color=ft.colors.BLUE_200)

        header = ft.Text("Welcome Back!", style=ft.TextThemeStyle.HEADLINE_LARGE, color=ft.colors.WHITE,
                         weight=ft.FontWeight.BOLD)
        subheader = ft.Text("Please login to your account", style=ft.TextThemeStyle.BODY_MEDIUM,
                            color=ft.colors.WHITE70)

        username = ft.TextField(
            label="Username",
            border_color=ft.colors.BLUE_200,
            focused_border_color=ft.colors.BLUE_400,
            prefix_icon=ft.icons.PERSON,
            bgcolor=ft.colors.WHITE10,
            color=ft.colors.WHITE,
        )
        password = ft.TextField(
            label="Password",
            password=True,
            border_color=ft.colors.BLUE_200,
            focused_border_color=ft.colors.BLUE_400,
            prefix_icon=ft.icons.LOCK,
            bgcolor=ft.colors.WHITE10,
            color=ft.colors.WHITE,
        )

        def login(e):
            user = self.app.db_manager.authenticate_user(username.value, password.value)
            if user:
                self.app.current_user = (user.id, user.username)
                session_token = self.app.db_manager.create_session(user.id, user.username)
                self.app.page.client_storage.set("session_token", session_token)
                self.app.show_snack_bar("Login successful!")
                self.app.show_main_menu()
            else:
                self.app.show_snack_bar("Invalid credentials!")

        login_button = ft.ElevatedButton(
            content=ft.Row([ft.Icon(ft.icons.LOGIN), ft.Text("Login")], alignment=ft.MainAxisAlignment.CENTER),
            on_click=login,
            style=ft.ButtonStyle(
                color=ft.colors.WHITE,
                bgcolor=ft.colors.BLUE_400,
                padding=15,
                shape=ft.RoundedRectangleBorder(radius=10),
            ),
            width=200,
        )

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
                subheader,
                ft.Container(height=20),
                username,
                password,
                ft.Container(height=10),
                login_button,
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
                    form
                ]),
                padding=30
            ),
            elevation=5,
            surface_tint_color=ft.colors.BLUE_GREY_800,
        )

        self.app.page.add(ft.Container(content=card, alignment=ft.alignment.center, expand=True))
        self.app.page.update()