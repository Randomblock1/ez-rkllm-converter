"""Configuration module for RKLLM converter with TUI and CLI support."""

import argparse
from typing import List, Dict
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Button, Input, Static, Label, Select
from textual.screen import Screen


# Platform-specific quantization types
PLATFORM_QTYPES = {
    "rk3576": ["w4a16", "w4a16_g32", "w4a16_g64", "w4a16_g128", "w8a8"],
    "rv1126b": ["w4a16", "w4a16_g32", "w4a16_g64", "w4a16_g128", "w8a8"],
    "rk3588": ["w8a8", "w8a8_g128", "w8a8_g256", "w8a8_g512"],
    "rk3562": ["w8a8", "w4a16_g32", "w4a16_g64", "w4a16_g128", "w4a8_g32"],
}

# Platform-specific NPU core options
PLATFORM_NPU_CORES = {
    "rk3576": [1, 2],
    "rv1126b": [1],
    "rk3588": [1, 2, 3],
    "rk3562": [1],
}


class Config:
    """Configuration holder for RKLLM converter."""
    
    def __init__(self):
        self.model_ids: List[str] = []
        self.platform: str = "rk3588"
        self.qtypes: List[str] = []
        self.hybrid_rates: List[str] = []
        self.optimizations: List[str] = []
        self.context_lengths: List[str] = []
        self.npu_cores: List[str] = []  # Can select multiple
    
    def get_default_npu_cores(self, platform: str) -> List[str]:
        """Get the default (highest) NPU cores for a platform."""
        cores = PLATFORM_NPU_CORES.get(platform, [3])
        return [str(max(cores))]
    
    def validate(self) -> tuple[bool, str]:
        """Validate the configuration."""
        if not self.model_ids:
            return False, "At least one model ID is required"
        
        if not self.qtypes:
            return False, "At least one quantization type is required"
        
        if not self.hybrid_rates:
            return False, "At least one hybrid rate is required"
        
        if not self.optimizations:
            return False, "At least one optimization level is required"
        
        if not self.context_lengths:
            return False, "At least one context length is required"
        
        if not self.npu_cores:
            return False, "At least one NPU core count is required"
        
        # Validate hybrid rates
        for rate in self.hybrid_rates:
            try:
                val = float(rate)
                if val < 0 or val > 1:
                    return False, f"Hybrid rate {rate} must be between 0 and 1"
            except ValueError:
                return False, f"Invalid hybrid rate: {rate}"
        
        # Validate qtypes for platform
        valid_qtypes = PLATFORM_QTYPES.get(self.platform, [])
        for qtype in self.qtypes:
            if qtype not in valid_qtypes:
                return False, f"Quantization type {qtype} not supported for platform {self.platform}"
        
        # Validate NPU cores for platform
        valid_cores = PLATFORM_NPU_CORES.get(self.platform, [])
        for core in self.npu_cores:
            try:
                core_int = int(core)
                if core_int not in valid_cores:
                    return False, f"NPU core count {core} not supported for platform {self.platform}. Valid options: {valid_cores}"
            except ValueError:
                return False, f"Invalid NPU core count: {core}"
        
        return True, ""
    
    def calculate_total_models(self) -> int:
        """Calculate the total number of model files that will be generated."""
        return (len(self.model_ids) * len(self.qtypes) * len(self.hybrid_rates) * 
                len(self.optimizations) * len(self.context_lengths) * len(self.npu_cores))


class ConfigScreen(Screen):
    """TUI screen for configuration."""
    
    CSS = """
    Screen {
        align: center middle;
    }
    
    Container {
        width: 80;
        height: auto;
        border: solid $accent;
        padding: 1;
    }
    
    Vertical {
        width: 100%;
        height: auto;
    }
    
    Label {
        margin-bottom: 1;
        color: $accent;
    }
    
    Input {
        margin-bottom: 1;
    }
    
    Static {
        margin-bottom: 1;
        color: $warning;
    }
    
    Horizontal {
        width: 100%;
        height: auto;
        align: center middle;
    }
    
    Button {
        margin: 1;
    }
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.error_message = ""
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            with Vertical():
                yield Label("RKLLM Converter Configuration")
                yield Static("Enter values separated by commas where multiple are allowed\n")
                
                yield Label("Model IDs (e.g., Qwen/Qwen3-4B-Thinking-2507):")
                yield Input(
                    value=",".join(self.config.model_ids),
                    placeholder="model/name,another/model",
                    id="model_ids"
                )
                
                yield Label("Platform:")
                yield Select(
                    options=[(k, k) for k in PLATFORM_QTYPES.keys()],
                    value=self.config.platform,
                    id="platform"
                )
                
                yield Label("NPU Cores (comma-separated):")
                yield Input(
                    value=",".join(self.config.npu_cores),
                    placeholder="1,2,3",
                    id="npu_cores"
                )
                yield Static(id="npu_cores_hint", markup=True)
                
                yield Label("Quantization Types (comma-separated):")
                yield Input(
                    value=",".join(self.config.qtypes),
                    placeholder="w8a8,w8a8_g128",
                    id="qtypes"
                )
                yield Static(id="qtypes_hint", markup=True)
                
                yield Label("Hybrid Rates (0.0-1.0, comma-separated):")
                yield Input(
                    value=",".join(self.config.hybrid_rates),
                    placeholder="0.0,0.2,0.4",
                    id="hybrid_rates"
                )
                
                yield Label("Optimizations (0 or 1, comma-separated):")
                yield Input(
                    value=",".join(self.config.optimizations),
                    placeholder="0,1",
                    id="optimizations"
                )
                
                yield Label("Context Lengths (up to 16k, e.g., 4k,8k,16k):")
                yield Input(
                    value=",".join(self.config.context_lengths),
                    placeholder="4k,8k,16k",
                    id="context_lengths"
                )
                
                yield Static(id="error_msg", markup=True)
                
                with Horizontal():
                    yield Button("Start Conversion", variant="primary", id="submit")
                    yield Button("Cancel", variant="error", id="cancel")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Update the qtypes hint when mounted."""
        self.update_qtypes_hint()
        self.update_npu_cores_hint()
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle platform selection changes."""
        if event.select.id == "platform":
            self.config.platform = str(event.value)
            self.update_qtypes_hint()
            self.update_npu_cores_hint()
    
    def update_qtypes_hint(self) -> None:
        """Update the hint for available qtypes based on platform."""
        qtypes = PLATFORM_QTYPES.get(self.config.platform, [])
        hint_widget = self.query_one("#qtypes_hint", Static)
        hint_widget.update(f"Available for {self.config.platform}: {', '.join(qtypes)}")
    
    def update_npu_cores_hint(self) -> None:
        """Update the hint for available NPU cores based on platform."""
        cores = PLATFORM_NPU_CORES.get(self.config.platform, [])
        hint_widget = self.query_one("#npu_cores_hint", Static)
        hint_widget.update(f"Available for {self.config.platform}: {', '.join(map(str, cores))} (default: {max(cores)})")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel":
            self.app.exit(result=None)
        elif event.button.id == "submit":
            self.submit_form()
    
    def submit_form(self) -> None:
        """Validate and submit the form."""
        # Get all values
        model_ids_input = self.query_one("#model_ids", Input).value
        platform_select = self.query_one("#platform", Select)
        npu_cores_input = self.query_one("#npu_cores", Input).value
        qtypes_input = self.query_one("#qtypes", Input).value
        hybrid_rates_input = self.query_one("#hybrid_rates", Input).value
        optimizations_input = self.query_one("#optimizations", Input).value
        context_lengths_input = self.query_one("#context_lengths", Input).value
        
        # Parse values
        self.config.model_ids = [m.strip() for m in model_ids_input.split(",") if m.strip()]
        self.config.platform = str(platform_select.value)
        self.config.npu_cores = list(set([n.strip() for n in npu_cores_input.split(",") if n.strip()]))
        self.config.qtypes = list(set([q.strip() for q in qtypes_input.split(",") if q.strip()]))
        self.config.hybrid_rates = list(set([h.strip() for h in hybrid_rates_input.split(",") if h.strip()]))
        self.config.optimizations = list(set([o.strip() for o in optimizations_input.split(",") if o.strip()]))
        self.config.context_lengths = list(set([c.strip() for c in context_lengths_input.split(",") if c.strip()]))
        
        # Validate
        valid, error = self.config.validate()
        if not valid:
            error_widget = self.query_one("#error_msg", Static)
            error_widget.update(f"[bold red]Error:[/] {error}")
            return
        
        # Exit with config
        self.app.exit(result=self.config)


class ConfigApp(App):
    """Textual app for configuration."""
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
    
    def on_mount(self) -> None:
        """Show the config screen on mount."""
        self.push_screen(ConfigScreen(self.config))


class ConfirmationScreen(Screen):
    """Confirmation screen to show total models and ask for confirmation."""
    
    CSS = """
    Screen {
        align: center middle;
    }
    
    Container {
        width: 80;
        height: auto;
        border: solid $accent;
        padding: 1;
    }
    
    Vertical {
        width: 100%;
        height: auto;
    }
    
    Label {
        margin-bottom: 1;
        color: $accent;
    }
    
    Static {
        margin-bottom: 1;
    }
    
    Horizontal {
        width: 100%;
        height: auto;
        align: center middle;
    }
    
    Button {
        margin: 1;
    }
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            with Vertical():
                yield Label("Configuration Summary")
                yield Static()
                
                total_models = self.config.calculate_total_models()
                
                yield Static(f"[bold]Models to process:[/] {len(self.config.model_ids)}")
                for model_id in self.config.model_ids:
                    yield Static(f"  • {model_id}")
                yield Static()
                
                yield Static(f"[bold]Platform:[/] {self.config.platform}")
                yield Static(f"[bold]NPU Cores:[/] {', '.join(self.config.npu_cores)}")
                yield Static(f"[bold]Quantization Types:[/] {', '.join(self.config.qtypes)}")
                yield Static(f"[bold]Hybrid Rates:[/] {', '.join(self.config.hybrid_rates)}")
                yield Static(f"[bold]Optimizations:[/] {', '.join(self.config.optimizations)}")
                yield Static(f"[bold]Context Lengths:[/] {', '.join(self.config.context_lengths)}")
                yield Static()
                
                yield Static(f"[bold yellow]Total model files to generate: {total_models}[/]")
                yield Static("[dim]Note: Each model will be uploaded immediately after generation.[/]")
                yield Static()
                
                yield Static(id="status_msg", markup=True)
                
                with Horizontal():
                    yield Button("Confirm & Start", variant="success", id="confirm")
                    yield Button("Edit Config", variant="primary", id="edit")
                    yield Button("Cancel", variant="error", id="cancel")
        
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel":
            self.app.exit(result=None)
        elif event.button.id == "edit":
            # Go back to config screen
            self.app.pop_screen()
        elif event.button.id == "confirm":
            # Confirm and proceed
            self.app.exit(result=self.config)


class ConfigWithConfirmationApp(App):
    """Textual app for configuration with confirmation."""
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.confirmed_config = None
    
    def on_mount(self) -> None:
        """Show the config screen on mount."""
        self.push_screen(ConfigScreen(self.config), self.check_config)
    
    def check_config(self, result) -> None:
        """Called when config screen returns."""
        if result is None:
            # User cancelled
            self.exit(result=None)
        else:
            # Show confirmation screen
            self.push_screen(ConfirmationScreen(result), self.handle_confirmation)
    
    def handle_confirmation(self, result) -> None:
        """Called when confirmation screen returns."""
        self.exit(result=result)


def parse_cli_args() -> Config:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to RKLLM format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive TUI mode (default)
  python main.py
  
  # Non-interactive mode with arguments
  python main.py --model-ids Qwen/Qwen3-4B-Thinking-2507 \\
                 --platform rk3588 \\
                 --qtypes w8a8,w8a8_g128 \\
                 --hybrid-rates 0.0,0.2,0.4 \\
                 --optimizations 0,1 \\
                 --context-lengths 4k,16k
        """
    )
    
    parser.add_argument(
        "--model-ids",
        type=str,
        help="Comma-separated list of HuggingFace model IDs"
    )
    
    parser.add_argument(
        "--platform",
        type=str,
        choices=list(PLATFORM_QTYPES.keys()),
        help="Target platform (rk3576, rv1126b, rk3588, rk3562)"
    )
    
    parser.add_argument(
        "--qtypes",
        type=str,
        help="Comma-separated list of quantization types"
    )
    
    parser.add_argument(
        "--hybrid-rates",
        type=str,
        help="Comma-separated list of hybrid rates (0.0-1.0)"
    )
    
    parser.add_argument(
        "--optimizations",
        type=str,
        help="Comma-separated list of optimization levels (0 or 1)"
    )
    
    parser.add_argument(
        "--context-lengths",
        type=str,
        help="Comma-separated list of context lengths (e.g., 4k,8k,16k)"
    )
    
    parser.add_argument(
        "--npu-cores",
        type=str,
        help="Comma-separated list of NPU core counts (e.g., 1,2,3)"
    )
    
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Run in non-interactive mode (requires all arguments)"
    )
    
    args = parser.parse_args()
    
    config = Config()
    
    # If any argument is provided, we're in CLI mode
    cli_mode = any([
        args.model_ids, args.platform, args.qtypes,
        args.hybrid_rates, args.optimizations, args.context_lengths,
        args.npu_cores, args.no_interactive
    ])
    
    if cli_mode:
        # Parse CLI arguments
        if args.model_ids:
            config.model_ids = [m.strip() for m in args.model_ids.split(",") if m.strip()]
        
        if args.platform:
            config.platform = args.platform
        
        if args.qtypes:
            config.qtypes = list(set([q.strip() for q in args.qtypes.split(",") if q.strip()]))
        
        if args.hybrid_rates:
            config.hybrid_rates = list(set([h.strip() for h in args.hybrid_rates.split(",") if h.strip()]))
        
        if args.optimizations:
            config.optimizations = list(set([o.strip() for o in args.optimizations.split(",") if o.strip()]))
        
        if args.context_lengths:
            config.context_lengths = list(set([c.strip() for c in args.context_lengths.split(",") if c.strip()]))
        
        if args.npu_cores:
            config.npu_cores = list(set([n.strip() for n in args.npu_cores.split(",") if n.strip()]))
        else:
            # Use default (highest) for the platform
            config.npu_cores = config.get_default_npu_cores(config.platform)
        
        # Validate
        valid, error = config.validate()
        if not valid:
            parser.error(error)
        
        # Show summary and ask for confirmation in CLI mode
        total = config.calculate_total_models()
        print(f"\n{'='*60}")
        print("Configuration Summary")
        print(f"{'='*60}")
        print(f"Models: {len(config.model_ids)}")
        for model_id in config.model_ids:
            print(f"  • {model_id}")
        print(f"Platform: {config.platform}")
        print(f"NPU Cores: {', '.join(config.npu_cores)}")
        print(f"Quantization Types: {', '.join(config.qtypes)}")
        print(f"Hybrid Rates: {', '.join(config.hybrid_rates)}")
        print(f"Optimizations: {', '.join(config.optimizations)}")
        print(f"Context Lengths: {', '.join(config.context_lengths)}")
        print(f"\nTotal model files to generate: {total}")
        print(f"{'='*60}")
        
        response = input("\nProceed with conversion? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Conversion cancelled.")
            exit(0)
        
        return config
    
    # Return None to indicate TUI mode should be used
    return None


def get_config() -> Config:
    """Get configuration either from CLI or TUI."""
    # Try CLI first
    config = parse_cli_args()
    
    if config is not None:
        # CLI mode
        return config
    
    # TUI mode - start with defaults
    config = Config()
    config.model_ids = ["Qwen/Qwen3-4B-Thinking-2507"]
    config.platform = "rk3588"
    config.qtypes = ["w8a8_g128", "w8a8"]
    config.hybrid_rates = ["0.0", "0.2", "0.4"]
    config.optimizations = ["0", "1"]
    config.context_lengths = ["4k", "16k"]
    config.npu_cores = config.get_default_npu_cores(config.platform)
    
    # Run TUI with confirmation
    app = ConfigWithConfirmationApp(config)
    result = app.run()
    
    if result is None:
        # User cancelled
        print("Configuration cancelled by user.")
        exit(0)
    
    return result
