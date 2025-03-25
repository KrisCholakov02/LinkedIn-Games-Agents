from time import sleep
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from src.web_automation.linkedin_game_agent import LinkedInGameAgent


def navigate_to_queens_puzzle(driver_path: str, username: str, password: str, headless: bool = False) -> LinkedInGameAgent:
    """
    Launches a LinkedInGameAgent instance, logs into LinkedIn, navigates to the Queens puzzle,
    and clicks the 'Solve puzzle' or 'Resume game' button if present.

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

    already_solved = False

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
        action_button = None
        fallback_button = None
        buttons = agent.driver.find_elements(By.TAG_NAME, "button")
        print(f"Found {len(buttons)} buttons. Searching for game state buttons...")

        for btn in buttons:
            try:
                span = btn.find_element(By.TAG_NAME, "span")
                text = span.text.strip()

                if "Solve puzzle" in text or "Resume game" in text:
                    action_button = btn
                    print("Found actionable game button:", text)
                    break
                elif "See results" in text:
                    fallback_button = btn  # store fallback if no action needed
            except Exception:
                continue

        if action_button:
            agent.driver.execute_script("arguments[0].scrollIntoView(true);", action_button)
            sleep(1)
            agent.driver.execute_script("arguments[0].click();", action_button)
            print("Clicked the 'Solve puzzle' or 'Resume game' button.")

        elif fallback_button:
            print("Puzzle already solved. 'See results' button is present. No need to run recognition.")
            already_solved = True
            return agent, already_solved

        else:
            print("No actionable puzzle button found (no Solve, Resume, or See results).")

    except Exception as e:
        print("An error occurred while handling puzzle action buttons:", e)

    return agent, already_solved