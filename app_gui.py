#!/usr/bin/env python3
"""
Depth Estimation Comparison App - Web GUI
==========================================

A Gradio-based web interface for comparing three depth estimation models:
- MiDaS
- Depth Anything V2
- Marigold

Usage:
    python app_gui.py

Then open your browser to the URL shown (usually http://localhost:7860)

Author: Claude Code
Date: 2026-01-09
"""

import os
import sys
import time
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    import gradio as gr
except ImportError:
    print("Error: Gradio is not installed.")
    print("Install it with: pip install gradio")
    sys.exit(1)

# Import the three depth estimation modules
try:
    from midas_depth import MiDaSDepthEstimator
    from depth_anything import DepthEstimator as DepthAnythingEstimator
    from marigold_depth import MarigoldDepthEstimator
except ImportError as e:
    print(f"Error importing depth estimation modules: {e}")
    print("Make sure all three depth estimation scripts are in the current directory.")
    sys.exit(1)


class DepthEstimationGUI:
    """Web GUI for depth estimation comparison."""

    def __init__(self):
        """Initialize the GUI with lazy model loading."""
        self.models = {
            'midas': None,
            'depth_anything': None,
            'marigold': None
        }
        self.model_loaded = {
            'midas': False,
            'depth_anything': False,
            'marigold': False
        }

    def load_model(self, model_name: str, **kwargs) -> str:
        """
        Load a specific model on demand.

        Args:
            model_name: Name of the model to load ('midas', 'depth_anything', 'marigold')
            **kwargs: Model-specific parameters

        Returns:
            Status message
        """
        if self.model_loaded[model_name]:
            return f"{model_name} is already loaded."

        try:
            if model_name == 'midas':
                model_type = kwargs.get('model_type', 'midas_v21_small_256')
                self.models['midas'] = MiDaSDepthEstimator(
                    model_type=model_type,
                    optimize=True,
                    device=None
                )
            elif model_name == 'depth_anything':
                model_size = kwargs.get('model_size', 'small')
                self.models['depth_anything'] = DepthAnythingEstimator(
                    model_size=model_size,
                    device=None
                )
            elif model_name == 'marigold':
                self.models['marigold'] = MarigoldDepthEstimator(
                    checkpoint="prs-eth/marigold-depth-v1-1",
                    device=None,
                    half_precision=True
                )

            self.model_loaded[model_name] = True
            return f"✓ {model_name} loaded successfully!"

        except Exception as e:
            return f"✗ Failed to load {model_name}: {str(e)}"

    def estimate_depth(
        self,
        image: np.ndarray,
        use_midas: bool,
        use_depth_anything: bool,
        use_marigold: bool,
        midas_model: str,
        da_model: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Estimate depth using selected models.

        Args:
            image: Input image as numpy array
            use_midas: Whether to run MiDaS
            use_depth_anything: Whether to run Depth Anything
            use_marigold: Whether to run Marigold
            midas_model: MiDaS model type
            da_model: Depth Anything model size

        Returns:
            Tuple of (midas_result, depth_anything_result, marigold_result, status_message)
        """
        if image is None:
            return None, None, None, "❌ Please upload an image first!"

        results = [None, None, None]
        status_messages = []
        timings = []

        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)

        # MiDaS
        if use_midas:
            try:
                if not self.model_loaded['midas']:
                    status_messages.append("Loading MiDaS...")
                    self.load_model('midas', model_type=midas_model)

                status_messages.append("Running MiDaS...")
                start = time.time()
                depth_map = self.models['midas'].estimate_depth(image)  # Pass numpy array
                elapsed = time.time() - start

                # Normalize and convert to RGB for display
                depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                depth_colored = plt.cm.viridis(depth_normalized)[:, :, :3]
                results[0] = (depth_colored * 255).astype(np.uint8)

                timings.append(f"MiDaS: {elapsed:.2f}s")
                status_messages.append("✓ MiDaS completed")
            except Exception as e:
                status_messages.append(f"✗ MiDaS failed: {str(e)}")

        # Depth Anything V2
        if use_depth_anything:
            try:
                if not self.model_loaded['depth_anything']:
                    status_messages.append("Loading Depth Anything V2...")
                    self.load_model('depth_anything', model_size=da_model)

                status_messages.append("Running Depth Anything V2...")
                start = time.time()
                depth_map = self.models['depth_anything'].estimate_depth(image)  # Pass numpy array
                elapsed = time.time() - start

                # Normalize and convert to RGB for display
                depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                depth_colored = plt.cm.Spectral_r(depth_normalized)[:, :, :3]
                results[1] = (depth_colored * 255).astype(np.uint8)

                timings.append(f"Depth Anything V2: {elapsed:.2f}s")
                status_messages.append("✓ Depth Anything V2 completed")
            except Exception as e:
                status_messages.append(f"✗ Depth Anything V2 failed: {str(e)}")

        # Marigold
        if use_marigold:
            try:
                if not self.model_loaded['marigold']:
                    status_messages.append("Loading Marigold...")
                    self.load_model('marigold')

                status_messages.append("Running Marigold...")
                start = time.time()
                depth_map, colored_depth = self.models['marigold'].estimate_depth(
                    pil_image,
                    ensemble_size=1,
                    seed=42
                )
                elapsed = time.time() - start

                # Normalize and convert to RGB for display
                depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                depth_colored = plt.cm.inferno(depth_normalized)[:, :, :3]
                results[2] = (depth_colored * 255).astype(np.uint8)

                timings.append(f"Marigold: {elapsed:.2f}s")
                status_messages.append("✓ Marigold completed")
            except Exception as e:
                status_messages.append(f"✗ Marigold failed: {str(e)}")

        # Create status message
        status = "\n".join(status_messages)
        if timings:
            status += "\n\n⏱️ Timing:\n" + "\n".join(timings)

        return results[0], results[1], results[2], status


def create_interface():
    """Create and configure the Gradio interface."""

    gui = DepthEstimationGUI()

    with gr.Blocks(title="Depth Estimation Comparison") as app:
        gr.Markdown("""
        # 🎯 Depth Estimation Comparison App

        Compare three state-of-the-art depth estimation models:
        - **MiDaS**: Robust monocular depth estimation
        - **Depth Anything V2**: Foundation model for depth estimation (NeurIPS 2024)
        - **Marigold**: Diffusion-based depth estimation (CVPR 2024)

        Upload an image and select which models to run!
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📤 Input")
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    height=400
                )

                gr.Markdown("### ⚙️ Settings")

                # Model selection
                with gr.Group():
                    gr.Markdown("**Select Models to Run:**")
                    use_midas = gr.Checkbox(label="MiDaS", value=True)
                    use_depth_anything = gr.Checkbox(label="Depth Anything V2", value=True)
                    use_marigold = gr.Checkbox(label="Marigold", value=True)

                # Model settings
                with gr.Accordion("Advanced Settings", open=False):
                    midas_model = gr.Dropdown(
                        choices=[
                            "midas_v21_small_256",
                            "midas_v21_384",
                            "dpt_hybrid_384",
                            "dpt_large_384"
                        ],
                        value="midas_v21_small_256",
                        label="MiDaS Model"
                    )

                    da_model = gr.Dropdown(
                        choices=["small", "base", "large"],
                        value="small",
                        label="Depth Anything Model Size"
                    )

                # Run button
                run_button = gr.Button(
                    "🚀 Estimate Depth",
                    variant="primary",
                    size="lg"
                )

                # Status output
                status_output = gr.Textbox(
                    label="Status",
                    lines=10,
                    max_lines=20,
                    interactive=False
                )

            with gr.Column(scale=2):
                # Output section
                gr.Markdown("### 📊 Results")

                with gr.Row():
                    midas_output = gr.Image(
                        label="MiDaS Output",
                        type="numpy",
                        height=300
                    )
                    da_output = gr.Image(
                        label="Depth Anything V2 Output",
                        type="numpy",
                        height=300
                    )

                with gr.Row():
                    marigold_output = gr.Image(
                        label="Marigold Output",
                        type="numpy",
                        height=300
                    )
                    gr.Markdown("""
                    ### 🎨 Color Mapping

                    - **MiDaS**: Viridis (purple → yellow)
                    - **Depth Anything V2**: Spectral (blue → red)
                    - **Marigold**: Inferno (black → yellow → white)

                    **Brighter colors = closer to camera**
                    **Darker colors = farther from camera**
                    """)

        # Connect the button to the function
        run_button.click(
            fn=gui.estimate_depth,
            inputs=[
                input_image,
                use_midas,
                use_depth_anything,
                use_marigold,
                midas_model,
                da_model
            ],
            outputs=[
                midas_output,
                da_output,
                marigold_output,
                status_output
            ]
        )

        # Examples
        gr.Markdown("### 📸 Example Images")
        gr.Examples(
            examples=[
                ["Depth-Anything-V2/assets/examples/demo01.jpg"] if os.path.exists("Depth-Anything-V2/assets/examples/demo01.jpg") else None,
                ["Depth-Anything-V2/assets/examples/demo02.jpg"] if os.path.exists("Depth-Anything-V2/assets/examples/demo02.jpg") else None,
            ],
            inputs=input_image,
            label="Click to load example"
        )

        # Footer
        gr.Markdown("""
        ---
        ### 📚 References

        - **MiDaS**: [GitHub](https://github.com/isl-org/MiDaS) | "Towards Robust Monocular Depth Estimation" (TPAMI 2022)
        - **Depth Anything V2**: [GitHub](https://github.com/DepthAnything/Depth-Anything-V2) | NeurIPS 2024
        - **Marigold**: [GitHub](https://github.com/prs-eth/Marigold) | CVPR 2024 (Oral, Best Paper Award Candidate)

        Built with ❤️ using Claude Code
        """)

    return app


def main():
    """Main entry point."""
    print("=" * 60)
    print("Depth Estimation Comparison App - Web GUI")
    print("=" * 60)
    print("\nStarting web server...")
    print("The app will open in your browser automatically.")
    print("If not, navigate to the URL shown below.\n")

    app = create_interface()

    # Launch the app
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create a public link
        inbrowser=True,  # Automatically open in browser
        theme=gr.themes.Soft()  # Theme moved here for Gradio 6.0
    )


if __name__ == "__main__":
    main()
