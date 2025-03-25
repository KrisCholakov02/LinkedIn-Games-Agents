from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep


class BaseGameAgent:
    def __init__(self, driver_path: str, headless: bool = False):
        self.driver_path = driver_path
        self.headless = headless
        self.driver = None
        self.wait = None

    def launch_driver(self):
        options = Options()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-gpu")

        service = Service(self.driver_path)
        self.driver = webdriver.Chrome(service=service, options=options)
        self.wait = WebDriverWait(self.driver, 20)

        print("[✓] Chrome driver launched.")

    def login(self, username: str, password: str):
        self.driver.get("https://www.linkedin.com/feed/")
        try:
            username_input = self.wait.until(EC.presence_of_element_located((By.ID, "username")))
            password_input = self.wait.until(EC.presence_of_element_located((By.ID, "password")))
            username_input.clear()
            password_input.clear()
            username_input.send_keys(username)
            password_input.send_keys(password)

            sign_in_button = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']")))
            sign_in_button.click()

            print("[✓] Login submitted.")
            sleep(5)
        except Exception as e:
            print("[✗] Login failed:", e)

    def navigate_to_game(self, game_name: str):
        """
        Finds and clicks the puzzle icon with given game name (e.g., 'Queens', 'Tange').

        Args:
            game_name (str): Visible alt text of the puzzle game icon
        """
        try:
            xpath = f"//img[@alt='{game_name}' and contains(@class, 'games-entrypoints-module__puzzle-icon')]"
            game_icon = self.wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
            game_icon.click()
            print(f"[✓] Clicked on the '{game_name}' game icon.")
        except Exception as e:
            print(f"[✗] Failed to find or click game icon for '{game_name}':", e)

    def quit(self):
        if self.driver:
            self.driver.quit()
            print("[✓] Browser closed.")