from src.core.base_game_agent import BaseGameAgent
from time import sleep
from selenium.webdriver.common.by import By

class LinkedInQueensAgent(BaseGameAgent):
    def __init__(self, driver_path, headless=False):
        super().__init__(driver_path, headless)
        self.launch_driver()

    def navigate_and_prepare(self, username: str, password: str) -> bool:
        self.login(username, password)
        self.navigate_to_game("Queens")
        sleep(5)
        return self._handle_game_state()

    def _handle_game_state(self):
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        for btn in buttons:
            try:
                span = btn.find_element(By.TAG_NAME, "span")
                text = span.text.strip()
                if "Solve puzzle" in text or "Resume game" in text:
                    btn.click()
                    print(f"[✓] Clicked '{text}' to start game.")
                    return True
                elif "See results" in text:
                    print("[✓] Puzzle already solved.")
                    return False
            except:
                continue
        print("[!] No valid game action found.")
        return False