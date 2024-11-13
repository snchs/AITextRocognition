# -*- coding: utf-8 -*-

import flet as ft  # Импортируем библиотеку Flet для создания интерфейсов
from database.db_manager import DatabaseManager  # Импортируем менеджер базы данных для работы с БД
from views.login_view import LoginView  # Импортируем представление для входа
from views.register_view import RegisterView  # Импортируем представление для регистрации
from views.text_scan_view import TextScanView  # Импортируем представление для сканирования текста
from views.history_view import HistoryView  # Импортируем представление для просмотра истории


class MedicalApp:
    def __init__(self, page: ft.Page):
        """Инициализация приложения, настройка страницы и проверка сессии пользователя."""
        self.page = page  # Сохраняем ссылку на объект страницы
        self.db_manager = DatabaseManager('scan_text_db.db')  # Инициализация менеджера базы данных
        self.setup_page()  # Настраиваем страницу приложения
        self.current_user = self.db_manager.check_session(page)  # Проверяем текущую сессию пользователя

    def setup_page(self):
        """Настройка параметров страницы приложения."""
        self.page.title = "TextScan AI"  # Установка заголовка окна
        self.page.window.width = 700  # Установка ширины окна
        self.page.window.height = 800  # Установка высоты окна
        self.page.window.center()  # Центрируем окно на экране
        self.page.theme_mode = ft.ThemeMode.DARK  # Установка темного режима темы
        self.page.padding = 50  # Установка внутреннего отступа страницы
        self.page.bgcolor = ft.colors.BLUE_GREY_900  # Установка фона страницы

    def show_snack_bar(self, message: str):
        """Отображение уведомления (snackbar) с заданным сообщением."""
        self.page.snack_bar = ft.SnackBar(content=ft.Text(message), bgcolor=ft.colors.BLUE_400)  # Создание snackbar
        self.page.snack_bar.open = True  # Открытие snackbar
        self.page.update()  # Обновление страницы для отображения изменений

    def show_login_view(self, e=None):
        """Отображение представления входа в систему."""
        LoginView(self).show()  # Создание и отображение представления входа

    def show_register_view(self, e=None):
        """Отображение представления регистрации."""
        RegisterView(self).show()  # Создание и отображение представления регистрации

    def show_textscan_view(self, e=None):
        """Отображение представления сканирования текста."""
        TextScanView(self).show()  # Создание и отображение представления сканирования текста

    def show_history_view(self, e=None):
        """Отображение представления истории сканирования."""
        HistoryView(self).show()  # Создание и отображение представления истории

    def logout(self, e=None):
        """Выход пользователя из системы."""
        self.current_user = None  # Сброс текущего пользователя
        self.page.client_storage.remove("session_token")  # Удаление токена сессии из клиентского хранилища
        self.show_snack_bar("You have been logged out.")  # Показ сообщения о выходе
        self.show_main_menu()  # Показ главного меню

    def show_main_menu(self, e=None):
        """Отображение главного меню приложения."""
        self.page.controls.clear()  # Очистка текущих элементов управления на странице

        # Иконка для представления текстового сканирования
        text_icon = ft.Icon(ft.icons.TEXT_SNIPPET_ROUNDED, size=80, color=ft.colors.BLUE_200)

        # Заголовок и подзаголовок главного меню
        header = ft.Text("Welcome to TextScan AI", style=ft.TextThemeStyle.HEADLINE_LARGE, color=ft.colors.WHITE,
                         weight=ft.FontWeight.BOLD)
        subheader = ft.Text("Handwriting recognition", style=ft.TextThemeStyle.BODY_MEDIUM,
                            color=ft.colors.WHITE70)
        username_text = ft.Text(f"", style=ft.TextThemeStyle.BODY_MEDIUM,
                                color=ft.colors.WHITE70)  # Текст для отображения имени пользователя

        # Определение элементов меню в зависимости от того, вошел ли пользователь
        if self.current_user:
            menu_items = [
                ft.ElevatedButton(
                    content=ft.Row([ft.Icon(ft.icons.DOCUMENT_SCANNER), ft.Text("Handwriting Recognition")],
                                   alignment=ft.MainAxisAlignment.CENTER),
                    on_click=self.show_textscan_view,  # Обработчик нажатия для перехода к сканированию
                    style=ft.ButtonStyle(color=ft.colors.WHITE, bgcolor=ft.colors.BLUE_400, padding=15,
                                         shape=ft.RoundedRectangleBorder(radius=10)),
                    width=250,
                ),
                ft.ElevatedButton(
                    content=ft.Row([ft.Icon(ft.icons.HISTORY), ft.Text("View History")],
                                   alignment=ft.MainAxisAlignment.CENTER),
                    on_click=self.show_history_view,  # Обработчик нажатия для просмотра истории
                    style=ft.ButtonStyle(color=ft.colors.WHITE, bgcolor=ft.colors.BLUE_400, padding=15,
                                         shape=ft.RoundedRectangleBorder(radius=10)),
                    width=250,
                ),
                ft.ElevatedButton(
                    content=ft.Row([ft.Icon(ft.icons.LOGOUT), ft.Text("Logout")],
                                   alignment=ft.MainAxisAlignment.CENTER),
                    on_click=self.logout,  # Обработчик нажатия для выхода из системы
                    style=ft.ButtonStyle(color=ft.colors.WHITE, bgcolor=ft.colors.RED_400, padding=15,
                                         shape=ft.RoundedRectangleBorder(radius=10)),
                    width=250,
                )
            ]

            # Отображение имени пользователя
            username_text = ft.Text(f"username: {self.current_user[1]}", style=ft.TextThemeStyle.BODY_MEDIUM,
                                    color=ft.colors.WHITE70)
        else:
            # Элементы меню для пользователей, не вошедших в систему
            menu_items = [
                ft.ElevatedButton(
                    content=ft.Row([ft.Icon(ft.icons.LOGIN), ft.Text("Login")], alignment=ft.MainAxisAlignment.CENTER),
                    on_click=self.show_login_view,  # Обработчик нажатия для входа
                    style=ft.ButtonStyle(color=ft.colors.WHITE, bgcolor=ft.colors.BLUE_400, padding=15,
                                         shape=ft.RoundedRectangleBorder(radius=10)),
                    width=250,
                ),
                ft.ElevatedButton(
                    content=ft.Row([ft.Icon(ft.icons.PERSON_ADD), ft.Text("Register")],
                                   alignment=ft.MainAxisAlignment.CENTER),
                    on_click=self.show_register_view,  # Обработчик нажатия для регистрации
                    style=ft.ButtonStyle(color=ft.colors.WHITE, bgcolor=ft.colors.BLUE_400, padding=15,
                                         shape=ft.RoundedRectangleBorder(radius=10)),
                    width=250,
                )
            ]

        # Создание макета меню с кнопками
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

        # Содержимое главного меню
        content = ft.Column(
            controls=[text_icon, header, subheader, username_text, ft.Container(height=30), menu],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=15,
        )

        # Создание карточки для отображения содержимого
        card = ft.Card(
            content=ft.Container(content=content, padding=30),
            elevation=5,
            surface_tint_color=ft.colors.BLUE_GREY_800,
        )

        # Добавление карточки на страницу
        self.page.add(
            ft.Container(
                content=card,
                alignment=ft.alignment.center,
                expand=True,
            )
        )
        self.page.update()  # Обновление страницы для отображения изменений

    def main(self):
        """Основной метод для запуска приложения."""
        if self.current_user:
            self.show_main_menu()  # Показать главное меню для вошедшего пользователя
        else:
            self.show_login_view()  # Показать представление входа для нового пользователя


def main(page: ft.Page):
    """Функция-обработчик для создания экземпляра приложения."""
    app = MedicalApp(page)  # Создание экземпляра приложения
    app.main()  # Запуск основного метода приложения


if __name__ == "__main__":
    # Запуск приложения, если этот файл выполняется как основной
    ft.app(target=main)
