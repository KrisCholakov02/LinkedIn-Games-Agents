from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options


class LinkedInGameAgent:
    def __init__(self, driver_path, headless=False):
        options = Options()
        if headless:
            options.add_argument("--headless=new")  # Use modern headless mode

        service = Service(driver_path)
        self.driver = webdriver.Chrome(service=service, options=options)

    def navigate_to_game(self, url):
        self.driver.get(url)

    def close(self):
        self.driver.quit()
