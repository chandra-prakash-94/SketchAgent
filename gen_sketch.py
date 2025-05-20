import argparse
import google.generativeai as genai
import ast
import cairosvg
import json
import os
import utils
import traceback

from dotenv import load_dotenv
from PIL import Image
from prompts import sketch_first_prompt, system_prompt, gt_example


def call_argparse():
    parser = argparse.ArgumentParser(description='Process Arguments')
    
    # General
    parser.add_argument('--concept_to_draw', type=str, default="cat")
    parser.add_argument('--seed_mode', type=str, default='deterministic', choices=['deterministic', 'stochastic'])
    parser.add_argument('--path2save', type=str, default=f"results/test")
    parser.add_argument('--model', type=str, default='gemini-1.5-pro-latest')
    parser.add_argument('--gen_mode', type=str, default='generation', choices=['generation', 'completion'])

    # Grid params
    parser.add_argument('--res', type=int, default=50, help="the resolution of the grid is set to 50x50")
    parser.add_argument('--cell_size', type=int, default=12, help="size of each cell in the grid")
    parser.add_argument('--stroke_width', type=float, default=7.0)

    args = parser.parse_args()
    args.grid_size = (args.res + 1) * args.cell_size

    args.save_name = args.concept_to_draw.replace(" ", "_")
    args.path2save = f"{args.path2save}/{args.save_name}"
    if not os.path.exists(args.path2save):
        os.makedirs(args.path2save)
        with open(f"{args.path2save}/experiment_log.json", 'w') as json_file:
            json.dump([], json_file, indent=4)
    return args


class SketchApp:
    def __init__(self, args):
        # General
        self.path2save = args.path2save
        self.target_concept = args.concept_to_draw

        # Grid related
        self.res = args.res
        self.num_cells = args.res
        self.cell_size = args.cell_size
        self.grid_size = (args.grid_size, args.grid_size)
        self.init_canvas, self.positions = utils.create_grid_image(res=args.res, cell_size=args.cell_size, header_size=args.cell_size)
        self.init_canvas_str = utils.image_to_str(self.init_canvas)
        self.cells_to_pixels_map = utils.cells_to_pixels(args.res, args.cell_size, header_size=args.cell_size)

        # SVG related 
        self.stroke_width = args.stroke_width
        
        # LLM Setup (you need to provide your ANTHROPIC_API_KEY in your .env file)
        self.cache = False
        self.max_tokens = 3000
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=google_api_key)
        self.client = genai.GenerativeModel(args.model)
        self.model = args.model
        self.input_prompt = sketch_first_prompt.format(concept=args.concept_to_draw, gt_sketches_str=gt_example)
        self.gen_mode = args.gen_mode
        self.seed_mode = args.seed_mode
        

    def call_llm(self, system_message, other_msg, additional_args):
        generation_config = {
            "max_output_tokens": self.max_tokens,
        }
        if "temperature" in additional_args:
            generation_config["temperature"] = additional_args["temperature"]
        if "top_k" in additional_args:
            generation_config["top_k"] = additional_args["top_k"]

        # For Gemini, the system message is passed as `system_instruction`
        # and messages are passed to `contents`
        # The caching functionality is not directly mapped here, assuming no direct equivalent for simplicity
        init_response = self.client.generate_content(
            contents=other_msg,
            generation_config=genai.types.GenerationConfig(**generation_config),
            system_instruction=system_message if isinstance(system_message, str) else system_message[0]['text'] # Handle both str and list cases for system_message
        )
        return init_response

    
    def define_input_to_llm(self, msg_history, init_canvas_str, msg):
        # Gemini expects messages in the format: [{'role': 'user'/'model', 'parts': [text/image]}]
        other_msg = []
        for message in msg_history:
            if isinstance(message['content'], list): # Handle cases where content is a list of dicts
                 parts = []
                 for item in message['content']:
                     if item['type'] == 'image':
                         parts.append({'mime_type': item['source']['media_type'], 'data': item['source']['data']})
                     elif item['type'] == 'text':
                         parts.append({'text': item['text']})
                 other_msg.append({'role': 'user' if message['role'] == 'user' else 'model', 'parts': parts})
            else: # Handle cases where content is a simple string
                 other_msg.append({'role': 'user' if message['role'] == 'user' else 'model', 'parts': [{'text': message['content']}]})


        current_content_parts = []
        if init_canvas_str is not None:
            # Assuming init_canvas_str is base64 encoded JPEG
            current_content_parts.append({"mime_type": "image/jpeg", "data": init_canvas_str})

        current_content_parts.append({"text": msg})
        
        # Add the current message to other_msg
        other_msg.append({"role": "user", "parts": current_content_parts})
        return other_msg
        

    def get_response_from_llm(
        self,
        msg,
        system_message,
        msg_history=[],
        init_canvas_str=None,
        prefill_msg=None,
        seed_mode="stochastic",
        stop_sequences=None,
        gen_mode="generation"
    ):  
        additional_args = {}
        if seed_mode == "deterministic":
            additional_args["temperature"] = 0.0
            additional_args["top_k"] = 1

        # Caching is not directly handled in Gemini's SDK in the same way as Anthropics's beta.
        # System message is handled by call_llm for Gemini.

        other_msg = self.define_input_to_llm(msg_history, init_canvas_str, msg)

        if gen_mode == "completion":
            if prefill_msg:
                # Gemini's completion equivalent is to have the prefill as the last part of the 'model' role message.
                # This is a simplified adaptation; true few-shot or completion might need more specific formatting.
                other_msg.append({"role": "model", "parts": [{"text": prefill_msg}]})
        
        # Stop sequences are part of generation_config in Gemini
        if stop_sequences:
            additional_args["stop_sequences"] = [stop_sequences]
        else:
            # Gemini uses 'stop_sequences' in generation_config, not directly in additional_args for the main call
            additional_args["stop_sequences"] = ["</answer>"]
            
        response = self.call_llm(system_message, other_msg, additional_args)
        
        # Accessing response text in Gemini
        try:
            content = response.text
        except Exception as e: # Added to catch potential errors with accessing response.text if it's not available.
            print(f"Error accessing response text: {e}")
            # Fallback or error handling if response.text is not directly available or parts need to be parsed
            content = "" 
            if response.parts:
                for part in response.parts:
                    if hasattr(part, 'text'):
                         content += part.text


        if gen_mode == "completion":
            # If there was a prefill, Gemini's response will be the completion part.
            # We need to combine it with the prefill_msg for the full content.
            content = f"{prefill_msg}{content}"
            if other_msg[-1]['role'] == 'model': # remove our added model prefill
                other_msg = other_msg[:-1]


        # saves to json - adapt structure for Gemini if needed
        if self.path2save is not None:
            # For Gemini, system_instruction is separate. History is a list of {'role': ..., 'parts': ...}
            # We'll save the system prompt and then the message history.
            log_data = [{"role": "system", "content": system_message if isinstance(system_message, str) else system_message[0]['text']}]
            
            # Adapt saved message history to a more generic format or keep as is if it's for logging purposes primarily
            # For now, keep it similar to the input structure for simplicity in logging
            # Convert 'parts' back to a simpler 'content' string for logging consistency if desired, or log 'parts' directly.
            adapted_history_for_logging = []
            for entry in other_msg:
                # Simplified logging: combine text parts. Image parts are not easily logged as simple text.
                text_content = "".join(part.get('text', '') for part in entry['parts'] if 'text' in part)
                adapted_history_for_logging.append({"role": entry["role"], "content": text_content})

            adapted_history_for_logging.append({
                "role": "model", # Gemini uses 'model' for assistant
                "content": content,
            })
            
            with open(f"{self.path2save}/experiment_log.json", 'w') as json_file:
                json.dump(log_data + adapted_history_for_logging, json_file, indent=4)
            print(f"Data has been saved to [{self.path2save}/experiment_log.json]")
        return content


    def call_model_for_sketch_generation(self):
        print("Calling LLM...")
        
        add_args = {}
        add_args["stop_sequences"] = f"</answer>" 

        msg_history = []
        init_canvas_str = None # self.init_canvas_str

        all_llm_output = self.get_response_from_llm(
            msg=self.input_prompt,
            system_message=system_prompt.format(res=self.res),
            msg_history=msg_history,
            init_canvas_str=init_canvas_str,
            seed_mode=self.seed_mode,
            gen_mode=self.gen_mode,
            **add_args
        )

        all_llm_output += f"</answer>"
        return all_llm_output
        

    def parse_model_to_svg(self, model_rep_sketch):
        # Parse model_rep with xml
        strokes_list_str, t_values_str = utils.parse_xml_string(model_rep_sketch, self.res)
        strokes_list, t_values = ast.literal_eval(strokes_list_str), ast.literal_eval(t_values_str)

        # extract control points from sampled lists
        all_control_points = utils.get_control_points(strokes_list, t_values, self.cells_to_pixels_map)

        # define SVG based on control point
        sketch_text_svg = utils.format_svg(all_control_points, dim=self.grid_size, stroke_width=self.stroke_width)
        return sketch_text_svg
        

    def generate_sketch(self):
        sketching_commands = self.call_model_for_sketch_generation()
        model_strokes_svg = self.parse_model_to_svg(sketching_commands)
        # saved the SVG sketch
        with open(f"{self.path2save}/{self.target_concept}.svg", "w") as svg_file:
            svg_file.write(model_strokes_svg)

        # vector->pixel 
        # save the sketch to png with blank backgournd
        cairosvg.svg2png(url=f"{self.path2save}/{self.target_concept}.svg", write_to=f"{self.path2save}/{self.target_concept}.png", background_color="white")
        
        # save the sketch to png on the canvas
        output_png_path = f"{self.path2save}/{self.target_concept}_canvas.png"
        cairosvg.svg2png(url=f"{self.path2save}/{self.target_concept}.svg", write_to=output_png_path)
        foreground = Image.open(output_png_path)
        self.init_canvas.paste(Image.open(output_png_path), (0, 0), foreground) 
        self.init_canvas.save(output_png_path)

        

# Initialize and run the SketchApp
if __name__ == '__main__':
    args = call_argparse()
    sketch_app = SketchApp(args)
    for attempts in range(3):
        try:
            sketch_app.generate_sketch()
            exit(0)
        except Exception as e:
            print(f"An error has occurred: {e}")
            traceback.print_exc()