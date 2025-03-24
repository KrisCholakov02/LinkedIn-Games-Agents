from time import sleep
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from src.web_automation.linkedin_game_agent import LinkedInGameAgent


def navigate_to_queens_puzzle(driver_path: str, username: str, password: str, headless: bool = False) -> LinkedInGameAgent:
    """
    Launches a LinkedInGameAgent instance, logs into LinkedIn, navigates to the Queens puzzle,
    and clicks the 'Solve puzzle' or 'Resume game' button.

    Args:
        driver_path (str): Path to the Chrome WebDriver executable.
        username (str): LinkedIn account username.
        password (str): LinkedIn account password.
        headless (bool): Whether to run the browser in headless mode. Default is False.

    Returns:
        LinkedInGameAgent: The agent with an active browser session.
    """
    agent = LinkedInGameAgent(driver_path=driver_path, headless=headless)
    agent.navigate_to_game("https://www.linkedin.com/feed/")
    print("Navigation successful. LinkedIn feed page should now be visible.")

    wait = WebDriverWait(agent.driver, 20)

    try:
        # Login
        username_input = wait.until(EC.presence_of_element_located((By.ID, "username")))
        password_input = wait.until(EC.presence_of_element_located((By.ID, "password")))
        username_input.clear()
        password_input.clear()
        username_input.send_keys(username)
        password_input.send_keys(password)

        sign_in_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']")))
        sign_in_button.click()
        print("Login submitted.")
    except Exception as e:
        print("An error occurred during login:", e)

    sleep(5)  # Wait for login to process

    try:
        queens_element = wait.until(EC.element_to_be_clickable((
            By.XPATH, "//img[@alt='Queens' and contains(@class, 'games-entrypoints-module__puzzle-icon')]"
        )))
        queens_element.click()
        print("Clicked on the Queens element.")
    except Exception as e:
        print("An error occurred while clicking the Queens element:", e)

    sleep(5)  # Wait for puzzle screen to load

    try:
        solve_button = None
        buttons = agent.driver.find_elements(By.TAG_NAME, "button")
        print(f"Found {len(buttons)} buttons. Searching for 'Solve puzzle' or 'Resume game'...")
        for btn in buttons:
            try:
                span = btn.find_element(By.TAG_NAME, "span")
                text = span.text.strip()
                if "Solve puzzle" in text or "Resume game" in text:
                    solve_button = btn
                    print("Found game action button with text:", text)
                    break
            except Exception:
                continue

        if solve_button:
            agent.driver.execute_script("arguments[0].scrollIntoView(true);", solve_button)
            sleep(1)
            agent.driver.execute_script("arguments[0].click();", solve_button)
            print("Clicked on the 'Solve puzzle' or 'Resume game' button.")
        else:
            print("Could not find the puzzle action button.")

    except Exception as e:
        print("An error occurred while handling puzzle action button:", e)

    return agent