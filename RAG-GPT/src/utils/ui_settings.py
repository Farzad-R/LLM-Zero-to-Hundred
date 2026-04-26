import gradio as gr


class UISettings:
    @staticmethod
    def toggle_sidebar(state: bool):
        state = not state
        return gr.update(visible=state), state

    @staticmethod
    def feedback(data: gr.LikeData) -> None:
        # In Gradio 5, data.value is a dict when messages use dict format
        value = data.value
        if isinstance(value, dict):
            content = value.get("value", str(value))
        else:
            content = str(value)
        action = "upvoted" if data.liked else "downvoted"
        print(f"You {action} this response: {content}")
