# Copyright (c) Meta Platforms, Inc. and affiliates.
from playwright.sync_api import sync_playwright


class WebArenaEditorException(Exception):
    """Exception raised for an error when editing the page.

    Attributes:
        message (str): The error message.
    """

    def __init__(self, message):
        """Initialize the WebArenaEditorException with a message.

        Args:
            message (str): The error message.
        """
        self.message = message
        super().__init__(self.message)


class BaseWebArenaEditor:
    """A base class to automate actions in a WebArena environment through a browser using Playwright."""

    def __init__(
        self,
        headless: bool = True,
    ) -> None:
        """Initialize the WebArenaEditor.

        Args:
            headless (bool, optional): Whether to run the browser in headless mode. Defaults to True.
        """
        self.headless = headless

    def __enter__(self):
        """Enter the runtime context related to the WebArenaEditor.

        Starts Playwright, launches the browser, and creates a new page.

        Returns:
            WebArenaEditor: The initialized editor instance.
        """
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)

        self.context = self.browser.new_context()

        self.page = self.context.new_page()
        return self

    def close(self):
        """Close the browser page, context, and the browser itself, as well as stop Playwright."""
        if hasattr(self, "page") and self.page:
            self.page.close()
        if hasattr(self, "context") and self.context:
            self.context.close()
        if hasattr(self, "browser") and self.browser:
            self.browser.close()
        if hasattr(self, "playwright") and self.playwright:
            self.playwright.stop()

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context, ensuring all resources are properly released."""
        self.close()

    def login(self, username: str, password: str) -> bool:
        raise NotImplementedError(
            "You must define the login function in the base classes"
        )

    def _simple_type_and_submit(
        self,
        url_of_action: str,
        selector_of_field_to_type_in: str,
        text_to_type: str,
        selector_of_submit_button: str,
    ):
        """Navigate to a page, fill in a field, and submit a form.

        Args:
            url_of_action (str): The URL of the page where the action will be performed.
            selector_of_field_to_type_in (str): CSS selector of the field to type text into.
            text_to_type (str): The text to be typed into the field.
            selector_of_submit_button (str): CSS selector of the button to submit the form.

        Raises:
            WebArenaEditorException: If the field to type into is not found.
        """
        self.page.goto(url_of_action, wait_until="networkidle")

        try:
            self.page.wait_for_selector(selector_of_field_to_type_in, timeout=10000)
        except:
            self.page.screenshot(path="debug_edit_submission.png")
            raise WebArenaEditorException(
                "[WebArenaEditor] Field to edit not found. Taking a screenshot for debugging."
            )

        self.page.fill(selector_of_field_to_type_in, text_to_type)
        self.page.click(selector_of_submit_button)
        self.page.wait_for_load_state("networkidle")