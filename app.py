# ==============================================================================
# Takumi - Artisan's Skill Inheritance System (English Version)
# ==============================================================================
# This notebook is intended for execution in Google Colab (L4 GPU).
# Objective: To fully translate the application into English for the competition
# submission, including UI, status messages, and AI prompts.
# ==============================================================================

# ------------------------------------------------------------------------------
# Step 1: Environment Setup and Library Installation
# ------------------------------------------------------------------------------
print("Step 1: Library Imports")
import gradio as gr
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import re
import time
import markdown2
from huggingface_hub import login
# from google.colab import userdata    <-Please import if you are in google colab
import subprocess
import math
from PIL import Image
import shutil

print("Installation complete.")
# ------------------------------------------------------------------------------
# Step 2: Login to Hugging Face
# ------------------------------------------------------------------------------
print("\nStep 2: Logging in to Hugging Face...")
try:
    # In Spaces, the token is set as an environment variable
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face.")
    else:
        print("HF_TOKEN not found. Running in read-only mode.")
except Exception as e:
    print(f"Failed to log in to Hugging Face. Error: {e}")

''' # If you are in google colab
print("\nStep 2: Logging in to Hugging Face...")
try:
    hf_token = userdata.get('HF_TOKEN')
    login(token=hf_token)
    print("Successfully logged in to Hugging Face.")
except Exception as e:
    print(f"Failed to log in to Hugging Face. Please check your Colab secrets settings. Error: {e}")
'''

# ------------------------------------------------------------------------------
# Step 3: Load Gemma 3n Model
# ------------------------------------------------------------------------------
print("\nStep 3: Loading Gemma 3n model...")
GEMMA_MODEL_ID = "google/gemma-3n-E4B-it"
processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
print("Model loading complete.")
# ------------------------------------------------------------------------------
# Step 4: Define Core Functions
# ------------------------------------------------------------------------------

def parse_report_to_html(report_text, is_log=False):
    """
    Parses the AI-generated report (Markdown format) into HTML with clickable timestamps.
    Applies special styling for logs if is_log is True.
    """
    def linkify_timestamps(match):
        # Supports both "XX min XX sec" and "XX分XX秒" for robustness
        time_str = match.group(0)
        minutes_match = re.search(r'(\d+)\s*(min|分)', time_str)
        seconds_match = re.search(r'(\d+)\s*(sec|秒)', time_str)
        minutes = int(minutes_match.group(1)) if minutes_match else 0
        seconds = int(seconds_match.group(1)) if seconds_match else 0
        total_seconds = minutes * 60 + seconds
        return f"<a href='#' class='timestamp-link' data-time='{total_seconds}' style='text-decoration: underline; color: #1d4ed8; font-weight: bold;'>{time_str}</a>"
    
    html_from_markdown = markdown2.markdown(report_text, extras=["fenced-code-blocks", "tables", "break-on-newline"])
    # Regex to find both English and Japanese time formats
    final_html = re.sub(r"(\d+)\s*(min|分)\s*(\d+)\s*(sec|秒)", linkify_timestamps, html_from_markdown)

    if is_log:
        return f"<div class='log-view'>{final_html}</div>"
    return final_html


def initial_analysis(video_path, frame_interval, audio_lang):
    """The main function implementing the 2-phase analysis approach."""
    
    if not video_path:
        yield "Please upload a video file.", gr.update(visible=False), None, None, None, None
        return

    yield "Starting analysis...", gr.update(visible=False), None, None, None, None
    
    # === Phase 1: Auditory Analysis (Timestamped Transcription + Environmental Sounds) ===
    yield "Step 1/4: Performing auditory analysis...", gr.update(visible=False), None, None, None, None
    
    audio_path = "/content/audio.wav"
    full_transcript = ""
    has_audio = False
    
    try:
        # Extract audio file
        subprocess.run([
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y", "-hide_banner", "-loglevel", "error"
        ], check=True)

        audio_chunk_dir = "/content/audio_transcript_chunks"
        if os.path.exists(audio_chunk_dir): shutil.rmtree(audio_chunk_dir)
        os.makedirs(audio_chunk_dir)

        dummy_image = Image.new('RGB', (100, 100), 'black')
        dummy_image_path = "/content/dummy_image.png"
        dummy_image.save(dummy_image_path)

        duration_str = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ]).decode("utf-8").strip()
        duration = float(duration_str)

        chunk_duration = 10
        num_chunks = math.ceil(duration / chunk_duration)

        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_path = os.path.join(audio_chunk_dir, f"chunk_{i}.wav")
            subprocess.run([
                "ffmpeg", "-i", audio_path, "-ss", str(start_time), "-t", str(chunk_duration),
                chunk_path, "-y", "-hide_banner", "-loglevel", "error"
            ], check=True)
            
            lang_instruction = "in Japanese" if audio_lang == "Japanese" else "in English"
            env_sound_instruction = "[sound of metal scraping]"
            
            messages = [{"role": "user", "content": [
                {"type": "text", "text": f"You are an expert in auditory analysis. Listen to this short audio clip and transcribe everything you hear {lang_instruction}. If you hear human speech, transcribe it accurately {lang_instruction}. If there is no speech but you hear important environmental sounds (e.g., machine operation sounds, tools hitting materials), describe the sound specifically (e.g., {env_sound_instruction}). Ignore the blank image."},
                {"type": "image", "url": dummy_image_path},
                {"type": "audio", "audio": chunk_path}
            ]}]
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    inputs[key] = value.to(model.dtype)
            
            output = model.generate(
                **inputs, max_new_tokens=512, 
                do_sample=True, top_k=50, top_p=0.9,
                disable_compile=True
            )
            transcript_part = processor.decode(output[0], skip_special_tokens=True).split("model\n")[-1].strip()
            
            minutes, seconds = divmod(start_time, 60)
            timestamp_str = f"[{int(minutes):02d} min {int(seconds):02d} sec]"
            full_transcript += f"{timestamp_str} {transcript_part}\n"
        
        has_audio = True

    except Exception as e:
        print(f"An error occurred during audio analysis: {e}")
        full_transcript = "Failed to analyze the audio track. The video may not contain audio."
        has_audio = False

    # === Phase 2: Integrated Analysis (Auditory Log + Video) ===
    yield "Step 2/4: Decomposing video into frames...", gr.update(visible=False), None, parse_report_to_html(full_transcript, is_log=True), None, None
    
    FRAME_DIR = "/content/frames"
    if os.path.exists(FRAME_DIR): shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR)
    
    subprocess.run(f'ffmpeg -i "{video_path}" -vf fps=1/{frame_interval} "{FRAME_DIR}/%04d.jpg" -hide_banner -loglevel error', shell=True, check=True)
    
    video_frames = []
    for frame_file in sorted(os.listdir(FRAME_DIR)):
        video_frames.append(os.path.join(FRAME_DIR, frame_file))

    yield "Step 3/4: Generating IKO Draft and questions...", gr.update(visible=False), None, parse_report_to_html(full_transcript, is_log=True), None, None

    prompt_text = f"""
You are an expert consultant specializing in knowledge transfer.
Here are the 'precise auditory logs (speech and environmental sounds)' and 'video (sequential frames)' of a certain task. Analyze both pieces of information comprehensively and perform the following two tasks for a junior technician to learn this task.

**[1] Create an IKO Draft:**
First, interpret the entire auditory log and video to divide the series of actions into meaningful **"Steps."** Then, create a report according to the IKO Format described below.
- **Crucial Rule:** For the "Reason (Why)," "Specific Tips (How)," and "Cautions" sections, **do NOT fill them with speculation if the information cannot be clearly inferred from the logs.** Instead, you must leave a placeholder: `[Supplemental information required]`

**[2] Generate Questions to Elicit Tacit Knowledge:**
Review the IKO Draft you created. For all items marked with `[Supplemental information required]`, generate up to **3 professional questions** to ask the expert technician to fill in these knowledge gaps.

---Auditory Log---
{full_transcript}
---End of Log---

**Output Format:**
### Initial Analysis Report (IKO Draft)
[Describe the draft report according to the IKO format here]

### Questions from AI
[List the questions here]

---
**IKO Format**

**Technical Report: [AI describes the task name]**

**1. Summary**
- **Key Safety Points:** [AI describes, or leaves placeholder]
- **Overall Workflow:** [AI describes]

**2. Interactive Procedure**
**Step 1: [AI describes the step name]**
- **Task (What):** [AI describes the action. **Must include the relevant timestamp (XX min XX sec) in the sentence.**]
- **Reason (Why):** `[Supplemental information required]`
- **Specific Tips (How):** `[Supplemental information required]`
- **Cautions:** `[Supplemental information required]`

[Repeat for all subsequent steps]
---
"""
    
    # ★★★ Error Fix: Create a dummy silent audio for Phase 2 ★★★
    dummy_audio_path = "/content/dummy_silent_audio.wav"
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1", 
        "-acodec", "pcm_s16le", dummy_audio_path, "-y", "-hide_banner", "-loglevel", "error"
    ], check=True)

    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt_text},
        {"type": "audio", "audio": dummy_audio_path} # Add dummy audio
    ]}]
    
    for i, frame_path in enumerate(video_frames):
        ts_sec = i * frame_interval
        mins, secs = divmod(ts_sec, 60)
        messages[0]["content"].append({"type": "text", "text": f"This is the frame at {int(mins):02d} min {int(secs):02d} sec into the video."})
        messages[0]["content"].append({"type": "image", "url": frame_path})

    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            inputs[key] = value.to(model.dtype)

    output = model.generate(
        **inputs, max_new_tokens=2048, 
        do_sample=True, top_k=50, top_p=0.9,
        disable_compile=True
    )
    report_and_questions = processor.decode(output[0], skip_special_tokens=True).split("model\n")[-1].strip()

    # 4. Generate HTML and update UI
    yield "Step 4/4: Displaying reports in the UI...", gr.update(visible=True), video_path, parse_report_to_html(full_transcript, is_log=True), parse_report_to_html(report_and_questions), "<script>playNotificationSound();</script>"


def generate_final_report(initial_report_and_questions_html, user_answers):
    yield "Generating final report...", gr.update(interactive=False), "", None
    initial_report_and_questions = re.sub('<[^<]+?>', '', initial_report_and_questions_html).replace("<br>", "\n")
    prompt = f"""
You are an Editor-in-Chief responsible for creating a final, flawless technical manual. Your source materials are a rough 'IKO Draft' and the definitive 'Expert's Answers.'

Your mission is to completely reconstruct the IKO Draft using the Expert's Answers as the absolute source of truth.

Your rules are:
1.  **Total Replacement:** For each step, completely replace the `[Supplemental information required]` sections with the relevant content from the Expert's Answers.
2.  **Intelligent Integration:** Do not simply append information. If the expert provides a critical safety warning, it **must** be moved to the 'Key Safety Points' summary and also integrated into the 'Cautions' of the relevant step.
3.  **Refine and Enhance:** Use the expert's detailed language to improve and enrich the initial 'Task (What)' descriptions.
4.  **No New Sections:** Do not create new, out-of-place sections at the end of the report. All information must be logically placed within the existing IKO structure.

---IKO Draft and AI Questions---
{initial_report_and_questions}
---End of Draft---

---Worker's Supplemental Response---
{user_answers}
---End of Response---
"""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    
    output = model.generate(
        **inputs, max_new_tokens=4096, 
        do_sample=True, top_k=50, top_p=0.9,
        disable_compile=True
    )
    final_report = processor.decode(output[0], skip_special_tokens=True).split("model\n")[-1].strip()
    final_html = parse_report_to_html(final_report)
    yield "Final report complete.", gr.update(interactive=True), final_html, "<script>playNotificationSound();</script>"

def transcribe_user_answer(audio_path, progress=gr.Progress()):
    """Transcribes the user's audio response into text."""
    if audio_path is None: return ""
    # This function would need to be implemented for the user response part
    # For now, it's a placeholder.
    return "User audio transcription feature not fully implemented in this version."

# ------------------------------------------------------------------------------
# Step 5: Build and Launch Gradio UI
# ------------------------------------------------------------------------------
print("\nStep 5: Building and launching the Gradio application...")

js_code = """
function setupEventListeners() {
    const body = document.body;
    if (body.dataset.listenerAttached === 'true') return;
    body.dataset.listenerAttached = 'true';

    body.addEventListener('click', function(event) {
        const link = event.target.closest('.timestamp-link');
        if (link) {
            event.preventDefault();
            const time = link.dataset.time;
            if (time) {
                const videos = document.querySelectorAll('video');
                videos.forEach(video => {
                    if (video.offsetParent !== null) {
                       video.currentTime = parseFloat(time);
                       video.play();
                    }
                });
            }
        }
    });
}

function playNotificationSound() {
    try {
        const context = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = context.createOscillator();
        const gain = context.createGain();
        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(523.25, context.currentTime); // C5
        gain.gain.setValueAtTime(0.3, context.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.00001, context.currentTime + 1);
        oscillator.connect(gain);
        gain.connect(context.destination);
        oscillator.start(context.currentTime);
        oscillator.stop(context.currentTime + 0.4);
    } catch (e) {
        console.error("Could not play sound:", e);
    }
}
"""
custom_css = """
#video_container {position: sticky; top: 1rem; align-self: flex-start;}
.log-view {
    background-color: #f5f5f5;
    border: 1px solid #e0e0e0;
    padding: 1rem;
    border-radius: 8px;
    font-family: monospace;
    white-space: pre-wrap;
    word-wrap: break-word;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# Takumi - Artisan's Skill Inheritance System")
    gr.Markdown("Upload a work video you want to analyze and press the 'Start Analysis' button.")
    
    sound_trigger = gr.HTML(visible=False)

    with gr.Row():
        with gr.Column(scale=1):
            video_upload = gr.Video(label="Upload Video File")
            frame_interval_slider = gr.Slider(
                minimum=1, 
                maximum=10, 
                step=1, 
                value=2, 
                label="Frame Extraction Interval (sec/frame)",
                info="Set how many seconds per frame to extract. A larger value speeds up analysis but reduces detail."
            )
            audio_language_dropdown = gr.Dropdown(["Japanese", "English"], value="English", label="Primary Language in Video")
            analyze_button = gr.Button("Start Analysis", variant="primary")
            status_text = gr.Textbox(label="Processing Status", interactive=False)
        
        with gr.Column(scale=2):
            with gr.Row(visible=False) as result_area:
                with gr.Column(scale=1, elem_id="video_container"):
                    video_display = gr.Video(label="Work Video", interactive=False)
                with gr.Column(scale=1):
                    with gr.Tabs() as tabs:
                        with gr.TabItem("Initial Analysis & Response", id=0):
                            initial_report_output = gr.HTML()
                            user_audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Your Supplemental Info (answer with voice)")
                            user_answer_box = gr.Textbox(label="AI Transcription Result (editable)", lines=5)
                            final_report_button = gr.Button("Generate Final Report with Response", variant="primary")
                        with gr.TabItem("Final Report", id=1):
                            final_report_output = gr.HTML()
                        with gr.TabItem("Auditory Analysis Log", id=2):
                            detailed_log_output = gr.HTML()

    analyze_button.click(
        fn=initial_analysis,
        inputs=[video_upload, frame_interval_slider, audio_language_dropdown],
        outputs=[status_text, result_area, video_display, detailed_log_output, initial_report_output, sound_trigger]
    )
    
    user_audio_input.stop_recording(
        fn=transcribe_user_answer,
        inputs=[user_audio_input],
        outputs=[user_answer_box]
    )

    final_report_button.click(
        fn=generate_final_report,
        inputs=[initial_report_output, user_answer_box],
        outputs=[status_text, final_report_button, final_report_output, sound_trigger]
    ).then(
        fn=lambda: gr.update(selected=1),
        inputs=None,
        outputs=tabs
    )
    
    demo.load(fn=None, inputs=None, outputs=None, js=f"() => {{ {js_code} setupEventListeners(); }}")

demo.launch()
# demo.launch(share=True, debug=True)  <-If you are in google colab.
