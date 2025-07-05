import logging
import os
import json
import smtplib
import requests
import base64
import pathlib
from datetime import datetime
from PIL import Image
import io
from together import Together
from typing import Optional
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from livekit.agents import function_tool, RunContext

logging.basicConfig(level=logging.INFO)


# ================================
# WEATHER TOOL
# ================================
@function_tool()
async def get_weather(context: RunContext, city: str) -> str:
    try:
        response = requests.get(f"https://wttr.in/{city}?format=3")
        if response.status_code == 200:
            logging.info(f"Weather for {city}: {response.text.strip()}")
            return response.text.strip()
        else:
            logging.error(
                f"Failed to get weather for {city}: {response.status_code}")
            return f"Could not retrieve weather for {city}."
    except Exception as e:
        logging.error(f"Error retrieving weather for {city}: {e}")
        return f"An error occurred while retrieving weather for {city}."


# ================================
# SEARCH TOOL (OpenRouter)
# ================================
@function_tool()
async def search_web(context: RunContext, query: str) -> str:
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "OpenRouter API key not found. Please check your .env file."

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "tngtech/deepseek-r1t-chimera:free",
            "messages": [{
                "role": "user",
                "content": query
            }],
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        logging.error(f"Error talking to OpenRouter AI: {e}")
        return f"Error talking to OpenRouter AI: {e}"


# ================================
# EMAIL TOOL
# ================================
@function_tool()
async def send_email(
    context: RunContext,
    to_email: str,
    subject: str,
    message: str,
    cc_email: Optional[str] = None,
) -> str:
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        gmail_user = os.getenv("GMAIL_USER")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")

        if not gmail_user or not gmail_password:
            logging.error(
                "Gmail credentials not found in environment variables")
            return "Email sending failed: Gmail credentials not configured."

        msg = MIMEMultipart()
        msg["From"] = gmail_user
        msg["To"] = to_email
        msg["Subject"] = subject

        recipients = [to_email]
        if cc_email:
            msg["Cc"] = cc_email
            recipients.append(cc_email)

        msg.attach(MIMEText(message, "plain"))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, recipients, msg.as_string())
        server.quit()

        logging.info(f"Email sent successfully to {to_email}")
        return f"Email sent successfully to {to_email}"

    except smtplib.SMTPAuthenticationError:
        logging.error("Gmail authentication failed")
        return "Email sending failed: Authentication error. Please check your Gmail credentials."
    except smtplib.SMTPException as e:
        logging.error(f"SMTP error occurred: {e}")
        return f"Email sending failed: SMTP error - {str(e)}"
    except Exception as e:
        logging.error(f"Error sending email: {e}")
        return f"An error occurred while sending email: {str(e)}"


# ================================
# AI IMAGE GENERATOR (Together AI)
# ================================
@function_tool()
async def generate_ai_image(context: RunContext, prompt: str) -> str:
    try:
        if not prompt or not isinstance(prompt, str) or len(
                prompt.strip()) < 3:
            return "❌ Please provide a clear description of the image you want."

        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            return "❌ Together AI API key not found. Please check your .env file."

        client = Together(api_key=api_key)

        response = client.images.generate(
            prompt=prompt.strip(),
            model="black-forest-labs/FLUX.1-schnell-Free",
            steps=4,
            n=1,
            height=1024,
            width=1024,
        )

        if not response or not hasattr(response, "data") or not response.data:
            logging.error(f"No image data returned: {response}")
            return "❌ No image data was returned. Try a different prompt."

        image_info = response.data[0]
        image_base64 = getattr(image_info, "b64_json", None)
        image_url = getattr(image_info, "url", None)

        if not image_base64 and not image_url:
            logging.error(f"No usable image returned: {vars(image_info)}")
            return "❌ Image generation failed: no usable output."

        if image_base64:
            image_bytes = base64.b64decode(image_base64)
            pictures_folder = os.path.join(pathlib.Path.home(), "Pictures")
            os.makedirs(pictures_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(pictures_folder,
                                      f"ai_image_{timestamp}.png")

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            data_url = f"data:image/png;base64,{image_base64}"

            return (
                f"✅ Image generated successfully!<br>"
                f"📁 Saved to: `{image_path}`<br>"
                f"🖼️ Preview:<br>"
                f'<img src="{data_url}" alt="{prompt}" style="max-width:100%; border-radius:12px;"/><br>'
                f'<a href="{data_url}" download="ai_image.png" style="display:inline-block;margin-top:10px;padding:8px 12px;background-color:#007bff;color:white;border-radius:6px;text-decoration:none;">⬇ Download Image</a>'
            )

        elif image_url:
            return (
                f"✅ Image generated successfully via URL!<br>"
                f"🖼️ Preview:<br>"
                f'<img src="{image_url}" alt="{prompt}" style="max-width:100%; border-radius:12px;"/><br>'
                f'<a href="{image_url}" download="ai_image.png" style="display:inline-block;margin-top:10px;padding:8px 12px;background-color:#007bff;color:white;border-radius:6px;text-decoration:none;">⬇ Download Image</a>'
            )

    except Exception as e:
        logging.error(f"Image generation error: {e}")
        return f"⚠️ Image generation failed: {str(e)}"


# ================================
# CODE GENERATION (OpenRouter)
# ================================
@function_tool()
async def generate_code(context: RunContext, prompt: str) -> str:
    """
    Generates code based on user prompt using OpenRouter with DeepSeek model.
    """
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "❌ Missing OpenRouter API key. Please check your .env file."

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-site.com",  # Optional
            "X-Title": "MiaAssistant",  # Optional
        }

        payload = {
            "model":
            "deepseek/deepseek-r1-0528:free",
            "messages": [{
                "role":
                "system",
                "content":
                "You are a professional senior software engineer. Write complete, well-documented code for the task."
            }, {
                "role": "user",
                "content": prompt
            }],
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=30)
        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        logging.error(f"Code generation failed: {e}")
        return f"❌ Code generation failed: {e}"


# ================================
# ESSAY WRITING (OpenRouter)
# ================================
@function_tool()
async def write_essay(context: RunContext,
                      topic: str,
                      words: int = 500) -> str:
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "❌ OpenRouter API key is missing. Please check your .env file."

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-site.com",
            "X-Title": "MiaAssistant",
        }

        system_prompt = (
            f"You are a professional academic writer. Write an original, human-sounding essay "
            f"on the topic provided. Avoid robotic phrasing, vary sentence structures, "
            f"and use natural transitions. Target a length of {words} words. "
            f"The tone should mimic a well-read human student or journalist.")

        user_prompt = f"Write an essay on: {topic}"

        payload = {
            "model":
            "mistralai/mixtral-8x7b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
            ],
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        logging.error(f"Essay generation failed: {e}")
        return f"❌ Essay generation failed: {e}"
