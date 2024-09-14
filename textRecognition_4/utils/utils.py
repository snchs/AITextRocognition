import flet as ft

def create_button(text, icon, on_click, width=250):
    return ft.ElevatedButton(
        content=ft.Row([ft.Icon(icon), ft.Text(text)],
                       alignment=ft.MainAxisAlignment.CENTER),
        on_click=on_click,
        style=ft.ButtonStyle(color=ft.colors.WHITE, bgcolor=ft.colors.BLUE_400, padding=15,
                             shape=ft.RoundedRectangleBorder(radius=10)),
        width=width,
    )

def create_card(content):
    return ft.Card(
        content=ft.Container(content=content, padding=20),
        elevation=5,
        surface_tint_color=ft.colors.BLUE_GREY_800,
    )