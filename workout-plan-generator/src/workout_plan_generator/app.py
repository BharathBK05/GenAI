
from __future__ import annotations

import os
from typing import Final

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from groq import APIConnectionError, APIStatusError, RateLimitError


load_dotenv()


def build_workout_prompt(
    fitness_goal: str,
    experience_level: str,
    days_per_week: int,
    equipment: list[str],
    injuries_or_limitations: str,
) -> str:
    """Build a detailed prompt from validated user inputs."""

    equipment_text = ", ".join(equipment)

    injury_text = injuries_or_limitations.strip() or "None reported"

    return f"""
Create a personalized, practical weekly workout plan.

USER PROFILE
- Fitness goal: {fitness_goal}
- Experience level: {experience_level}
- Available training days: exactly {days_per_week} days per week
- Available equipment: {equipment_text}
- Injuries or physical limitations: {injury_text}

MANDATORY INSTRUCTIONS
1. Create exactly {days_per_week} training days, labelled Day 1
   through Day {days_per_week}.
2. Make the plan appropriate for the user's experience level.
3. Use only the listed equipment. Do not assume gym machines,
   barbells, or other unavailable equipment.
4. Account for the user's stated limitations. Avoid exercises
   that directly conflict with those limitations.
5. Include a warm-up, exercises, sets, repetitions or duration,
   rest periods, and a cool-down for each training day.
6. Give a clear progression strategy appropriate to the goal.
7. Include rest or recovery guidance for non-training days.
8. Explain briefly why the weekly structure suits the goal.
9. Use Markdown headings, bullet points, and tables where useful.
10. Do not invent medical diagnoses or claim to treat injuries.
11. If limitations are reported, include a short disclaimer
    recommending professional guidance when appropriate.
12. If information is insufficient to prescribe an exercise
    safely, choose a conservative alternative or explain the
    limitation rather than guessing.

OUTPUT FORMAT
## Overview
Summarize the goal, training split, and progression approach.

## Day 1
Include warm-up and a table with Exercise, Sets, Reps/Duration,
and Rest. Add brief form guidance when useful.

Continue through exactly Day {days_per_week}.

## Recovery and Progression
Explain rest days, progression, and when to reduce intensity.

## Safety Note
Include a concise safety disclaimer, especially if limitations
were reported.

Make the plan actionable and specific, not generic.
"""


def generate_workout_plan(
    fitness_goal: str,
    experience_level: str,
    days_per_week: int,
    equipment: list[str],
    injuries_or_limitations: str = "",
) -> str:
    """Validate inputs, call Groq, and return the workout plan."""

    valid_goals = {
        "Build muscle",
        "Lose fat",
        "General fitness",
        "Improve endurance",
    }
    valid_levels = {"Beginner", "Intermediate", "Advanced"}
    valid_equipment = {
        "No equipment",
        "Home dumbbells",
        "Full gym",
    }

    if fitness_goal not in valid_goals:
        raise ValueError("Please select a valid fitness goal.")

    if experience_level not in valid_levels:
        raise ValueError("Please select a valid experience level.")

    if not isinstance(days_per_week, int) or not 1 <= days_per_week <= 7:
        raise ValueError("Training days must be between 1 and 7.")

    if not equipment or not set(equipment).issubset(valid_equipment):
        raise ValueError("Please select valid equipment options.")

    # Prevent incompatible equipment selections.
    if "No equipment" in equipment and len(equipment) > 1:
        raise ValueError(
            "'No equipment' cannot be combined with other options."
        )

    api_key = os.getenv("GROQ_API_KEY")
    MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    if not api_key:
        raise RuntimeError(
            "Groq API key is missing. Add GROQ_API_KEY to your .env file."
        )

    prompt = build_workout_prompt(
        fitness_goal=fitness_goal,
        experience_level=experience_level,
        days_per_week=days_per_week,
        equipment=equipment,
        injuries_or_limitations=injuries_or_limitations,
    )

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful, evidence-informed fitness "
                        "programming assistant. Follow every user "
                        "constraint. Do not provide medical diagnoses. "
                        "Return a structured, practical Markdown plan."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_completion_tokens=4000,
        )

        if not response.choices:
            raise ValueError("The model returned no choices.")

        content = response.choices[0].message.content

        if not isinstance(content, str) or not content.strip():
            raise ValueError("The model returned an empty response.")

        # Basic sanity check: a plan should have some structure.
        if "day" not in content.lower() or len(content.strip()) < 100:
            raise ValueError(
                "The model response does not look like a valid plan."
            )

        return content.strip()

    except RateLimitError as exc:
        raise RuntimeError(
            "The API rate limit was reached. Please wait and try again."
        ) from exc

    except APIConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Groq. Check your internet connection."
        ) from exc

    except APIStatusError as exc:
        if exc.status_code in (401, 403):
            message = "Groq rejected the API key. Check your credentials."
        elif exc.status_code == 429:
            message = "Groq rate limit or quota reached. Try again later."
        elif exc.status_code >= 500:
            message = "Groq is temporarily unavailable. Try again later."
        else:
            message = f"Groq returned an API error ({exc.status_code})."

        raise RuntimeError(message) from exc

    except ValueError:
        raise

    except Exception as exc:
        raise RuntimeError(
            "Something went wrong while generating the plan. "
            "Please try again."
        ) from exc


def main() -> None:
    """Render the Streamlit user interface."""

    st.set_page_config(
        page_title="Workout Plan Generator",
        page_icon="💪",
        layout="wide",
    )

    st.title("💪 Personalized Workout Plan Generator")
    st.write(
        "Tell us about your fitness goals and available equipment. "
        "We'll create a structured weekly plan using AI."
    )

    with st.form("workout_form"):
        st.subheader("Your fitness profile")

        fitness_goal = st.selectbox(
            "Fitness goal",
            [
                "Build muscle",
                "Lose fat",
                "General fitness",
                "Improve endurance",
            ],
        )

        experience_level = st.selectbox(
            "Experience level",
            ["Beginner", "Intermediate", "Advanced"],
        )

        days_per_week = st.slider(
            "Training days per week",
            min_value=1,
            max_value=7,
            value=3,
        )

        equipment = st.multiselect(
            "Available equipment",
            [
                "No equipment",
                "Home dumbbells",
                "Full gym",
            ],
            default=["No equipment"],
        )

        injuries_or_limitations = st.text_area(
            "Injuries or limitations (optional)",
            placeholder=(
                "For example: knee discomfort, limited shoulder mobility, "
                "or no overhead pressing."
            ),
            help=(
                "Describe relevant limitations. The app cannot diagnose "
                "conditions or guarantee that an exercise is safe."
            ),
        )

        submitted = st.form_submit_button(
            "Generate Workout Plan",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not equipment:
            st.error("Please select at least one equipment option.")
        elif (
            "No equipment" in equipment
            and len(equipment) > 1
        ):
            st.error(
                "'No equipment' cannot be combined with other options."
            )
        else:
            try:
                with st.spinner("Creating your personalized plan..."):
                    plan = generate_workout_plan(
                        fitness_goal=fitness_goal,
                        experience_level=experience_level,
                        days_per_week=days_per_week,
                        equipment=equipment,
                        injuries_or_limitations=injuries_or_limitations,
                    )

                st.session_state["workout_plan"] = plan
                st.session_state["workout_profile"] = {
                    "goal": fitness_goal,
                    "level": experience_level,
                    "days": days_per_week,
                    "equipment": equipment,
                }

            except ValueError as exc:
                st.error(f"Invalid input: {exc}")

            except RuntimeError as exc:
                st.error(str(exc))

    # Display the saved plan even when Streamlit reruns the page.
    plan = st.session_state.get("workout_plan")

    if plan:
        st.divider()
        st.subheader("Your weekly workout plan")

        profile = st.session_state.get("workout_profile", {})
        st.caption(
            f"Goal: {profile.get('goal', 'Not specified')} | "
            f"Level: {profile.get('level', 'Not specified')} | "
            f"Days: {profile.get('days', 'Not specified')}"
        )

        st.markdown(plan)

        st.download_button(
            label="Download plan as Markdown",
            data=plan,
            file_name="weekly_workout_plan.md",
            mime="text/markdown",
        )

        if st.button("Regenerate plan"):
            try:
                with st.spinner("Generating another variation..."):
                    new_plan = generate_workout_plan(
                        fitness_goal=profile["goal"],
                        experience_level=profile["level"],
                        days_per_week=profile["days"],
                        equipment=profile["equipment"],
                        injuries_or_limitations=injuries_or_limitations,
                    )

                st.session_state["workout_plan"] = new_plan
                st.rerun()

            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
