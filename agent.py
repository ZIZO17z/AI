import logging
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import noise_cancellation, google
from prompts import AGENT_INSTRUCTION, SESSION_INSTRUCTION
from tools import get_weather, search_web, send_email, generate_ai_image, generate_code, write_essay

load_dotenv()  # Load environment variables early


# ================================
# AGENT DEFINITION
# ================================
class Assistant(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions=AGENT_INSTRUCTION,
            llm=google.beta.realtime.RealtimeModel(
                voice="Aoede",  # Voice options: Aoede, Melody, etc.
                temperature=0.8,
            ),
            tools=[
                get_weather, search_web, send_email, generate_ai_image,
                generate_code, write_essay
            ],
        )


# ================================
# ENTRYPOINT FUNCTION
# ================================
async def entrypoint(ctx: agents.JobContext):
    # Connect to LiveKit room first
    await ctx.connect()

    session = AgentSession()

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            video_enabled=True,
            noise_cancellation=noise_cancellation.BVC(
            ),  # LiveKit BVC for voice clarity
        ),
    )

    await session.generate_reply(instructions=SESSION_INSTRUCTION, )


# ================================
# MAIN LAUNCHER
# ================================
if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
