#!/usr/bin/env python3
"""
Create a GitHub release with trained model checkpoints and sample outputs.

This script:
1. Finds the latest regression and diffusion checkpoints
2. Extracts model configuration details from YAML files
3. Packages checkpoints and sample plots
4. Creates a GitHub release with detailed release notes

Requirements:
    pip install PyGithub PyYAML

Usage:
    # Set environment variable first:
    export GITHUB_TOKEN=your_github_token
    
    # Then run:
    python create_github_release.py --tag v1.0.0 --name "NZ GEFS-WHACS Models v1.0.0"
    
    # Or use gh CLI (preferred):
    python create_github_release.py --tag v1.0.0 --name "NZ GEFS-WHACS Models v1.0.0" --use-gh-cli
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Install with: pip install PyYAML")
    sys.exit(1)


class CheckpointInfo:
    """Container for checkpoint information."""
    
    def __init__(self, path: str, checkpoint_type: str):
        self.path = path
        self.checkpoint_type = checkpoint_type  # 'regression' or 'diffusion'
        self.filename = os.path.basename(path)
        self.size_mb = os.path.getsize(path) / (1024 * 1024)
        self.modified_time = datetime.fromtimestamp(os.path.getmtime(path))
        
        # Extract iteration number from filename
        match = re.search(r'\.0\.(\d+)\.mdlus$', self.filename)
        self.iteration = int(match.group(1)) if match else 0


def find_latest_checkpoint(checkpoint_dir: str, pattern: str = "*.mdlus") -> Optional[str]:
    """Find the latest checkpoint file in a directory."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not checkpoints:
        return None
    
    # Sort by modification time, newest first
    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return checkpoints[0]


def load_config_info(config_path: str) -> Dict:
    """Extract relevant information from YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    info = {
        'model_size': config.get('defaults', [{}])[2] if len(config.get('defaults', [])) > 2 else 'unknown',
        'training_duration': config.get('training', {}).get('hp', {}).get('training_duration', 'N/A'),
        'batch_size': config.get('training', {}).get('hp', {}).get('total_batch_size', 'N/A'),
        'batch_size_per_gpu': config.get('training', {}).get('hp', {}).get('batch_size_per_gpu', 'N/A'),
        'checkpoint_level': config.get('training', {}).get('perf', {}).get('songunet_checkpoint_level', 'N/A'),
        'dataloader_workers': config.get('training', {}).get('perf', {}).get('dataloader_workers', 'N/A'),
        'input_variables': config.get('validation', {}).get('input_variables', []),
        'output_variables': config.get('validation', {}).get('output_variables', []),
        'target_resolution': config.get('validation', {}).get('target_resolution', 'N/A'),
    }
    
    # Extract model_size string if it's a dict
    if isinstance(info['model_size'], dict):
        info['model_size'] = info['model_size'].get('model_size', 'unknown')
    elif isinstance(info['model_size'], str) and info['model_size'].startswith('model_size:'):
        info['model_size'] = info['model_size'].split(':')[-1].strip()
    
    return info


def find_sample_plots(plots_dir: str, max_plots: int = 6) -> List[str]:
    """Find sample plots to include in the release."""
    sample_plots = glob.glob(os.path.join(plots_dir, "*.sample.png"))
    sample_plots.sort()
    return sample_plots[:max_plots]


def generate_release_notes(
    regression_info: CheckpointInfo,
    diffusion_info: CheckpointInfo,
    regression_config: Dict,
    diffusion_config: Dict,
    plots: List[str],
    tag: str,
    repo: str
) -> str:
    """Generate comprehensive release notes."""
    
    notes = f"""# NZ GEFS-WHACS Wave Downscaling Models

This release contains trained models for downscaling GEFS wave forecasts (0.25°) to WHACS resolution (~0.0625°) over New Zealand waters.

## 📦 Release Contents

### Checkpoints
- **Regression Model**: `{regression_info.filename}` ({regression_info.size_mb:.1f} MB)
  - Iteration: {regression_info.iteration:,}
  - Modified: {regression_info.modified_time.strftime('%Y-%m-%d %H:%M:%S')}

- **Diffusion Model**: `{diffusion_info.filename}` ({diffusion_info.size_mb:.1f} MB)
  - Iteration: {diffusion_info.iteration:,}
  - Modified: {diffusion_info.modified_time.strftime('%Y-%m-%d %H:%M:%S')}

### Sample Outputs
{len(plots)} sample visualization plots showing model predictions (see below)

---

## 🎨 Sample Visualizations

Below are sample predictions from the diffusion model showing downscaled wave parameters:

"""
    
    # Add inline images for each plot
    for plot in plots:
        plot_filename = os.path.basename(plot)
        # Extract timestamp from filename (e.g., "2023-01-01T00:00:00" from "2023-01-01T00:00:00.sample.png")
        timestamp = plot_filename.replace('.sample.png', '')
        
        # GitHub sanitizes filenames: colons (:) become periods (.)
        # So "2023-01-01T00:00:00.sample.png" becomes "2023-01-01T00.00.00.sample.png"
        sanitized_filename = plot_filename.replace(':', '.')
        
        plot_url = f"https://github.com/{repo}/releases/download/{tag}/{sanitized_filename}"
        notes += f"### {timestamp}\n\n"
        notes += f"![{timestamp} prediction]({plot_url})\n\n"
    
    notes += "---\n\n"
    
    notes += f"""

## 🔧 Model Configuration

### Regression Model
- **Model Size**: `{regression_config['model_size']}`
- **Training Duration**: {regression_config['training_duration']:,} samples
- **Batch Size**: {regression_config['batch_size']} (per GPU: {regression_config['batch_size_per_gpu']})
- **Gradient Checkpointing Level**: {regression_config['checkpoint_level']}
- **DataLoader Workers**: {regression_config['dataloader_workers']}

### Diffusion Model
- **Model Size**: `{diffusion_config['model_size']}`
- **Training Duration**: {diffusion_config['training_duration']:,} samples
- **Batch Size**: {diffusion_config['batch_size']} (per GPU: {diffusion_config['batch_size_per_gpu']})
- **Gradient Checkpointing Level**: {diffusion_config['checkpoint_level']}
- **DataLoader Workers**: {diffusion_config['dataloader_workers']}

---

## 📊 Dataset Details

### Input Variables (GEFS)
{', '.join(f'`{v}`' for v in regression_config['input_variables'])}

### Output Variables (WHACS)
{', '.join(f'`{v}`' for v in regression_config['output_variables'])}

### Resolution
- **Input**: 0.25° (GEFS)
- **Output**: {regression_config['target_resolution']}° (WHACS)
- **Upsampling Factor**: 4x

---

## 🚀 Usage

### Loading Checkpoints

```python
from physicsnemo.deploy import Module

# Load regression model
regression_model = Module.from_checkpoint(
    "checkpoints_regression/{regression_info.filename}"
)

# Load diffusion model
diffusion_model = Module.from_checkpoint(
    "checkpoints_diffusion/{diffusion_info.filename}"
)
```

### Running Inference

```bash
# Generate predictions
python examples/weather/corrdiff/generate.py \\
    --config-name=config_generate_gefs_WHACS \\
    regression_checkpoint_path=checkpoints_regression/{regression_info.filename} \\
    diffusion_checkpoint_path=checkpoints_diffusion/{diffusion_info.filename}
```

---

## 📈 Training Details

Both models were trained on GEFS wave reforecast data paired with WHACS observations over New Zealand waters:
- **Training Years**: 2000-2020
- **Validation Years**: 2021-2023
- **Domain**: New Zealand coastal and offshore waters
- **Variables**: Significant wave height (`hs`), wave direction (`dir`), mean wave period (`t01`)

The regression model provides the base downscaled prediction, while the diffusion model adds realistic fine-scale details.

---

## 📝 Citation

If you use these models, please cite:

```bibtex
@software{{physicsnemo_nz_whacs,
  title = {{NZ GEFS-WHACS Wave Downscaling Models}},
  author = {{NVIDIA PhysicsNeMo Team}},
  year = {{{datetime.now().year}}},
  url = {{https://github.com/NVIDIA/physicsnemo}}
}}
```

---

## 🐛 Known Issues & Notes

- Models are optimized for NZ domain only
- Ensure all dependencies from `requirements.txt` are installed
- GPU with at least 16GB VRAM recommended for inference

---

## 📧 Contact

For questions or issues, please open a GitHub issue or consult the documentation.
"""
    
    return notes


def check_and_create_git_tag(tag: str, create_tag: bool = True, push_tag: bool = True) -> bool:
    """Check if git tag exists, create it, and push to remote if needed."""
    
    # Check if in a git repository
    try:
        subprocess.run(['git', 'rev-parse', '--git-dir'], 
                      check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Warning: Not in a git repository. Tag may not be created properly.")
        return False
    
    # Check if tag already exists locally
    result = subprocess.run(['git', 'tag', '-l', tag], 
                          capture_output=True, text=True)
    
    tag_exists_locally = tag in result.stdout
    
    if tag_exists_locally:
        print(f"✓ Git tag '{tag}' already exists locally")
    elif create_tag:
        # Create the tag
        print(f"Creating git tag '{tag}'...")
        try:
            subprocess.run(['git', 'tag', '-a', tag, '-m', f'Release {tag}'], 
                          check=True, capture_output=True)
            print(f"✓ Git tag '{tag}' created locally")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating git tag: {e}")
            return False
    else:
        print(f"⚠️  Warning: Git tag '{tag}' does not exist locally")
        return False
    
    # Push tag to remote (CRITICAL for gh release create to work properly)
    if push_tag:
        print(f"Pushing tag '{tag}' to remote...")
        try:
            result = subprocess.run(['git', 'push', 'origin', tag], 
                                  capture_output=True, text=True)
            if "already exists" in result.stderr.lower():
                print(f"✓ Tag '{tag}' already exists on remote")
            else:
                print(f"✓ Tag '{tag}' pushed to remote")
        except subprocess.CalledProcessError as e:
            # Try to get more info about the error
            print(f"⚠️  Warning: Could not push tag to remote: {e}")
            print(f"   You may need to push it manually: git push origin {tag}")
            print(f"   The release will be created, but images may not display correctly")
            print()
    
    return True


def create_release_with_gh_cli(
    tag: str,
    name: str,
    notes: str,
    files: List[str],
    repo: str = "uoa-eresearch/physicsnemo",
    draft: bool = False,
    prerelease: bool = False,
    create_tag: bool = True
) -> None:
    """Create release using gh CLI (recommended method)."""
    
    print("Creating release with gh CLI...")
    
    # Check if gh CLI is installed
    try:
        subprocess.run(['gh', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: gh CLI not found. Install from https://cli.github.com/")
        sys.exit(1)
    
    # Check/create git tag AND PUSH IT (critical!)
    if create_tag:
        tag_ok = check_and_create_git_tag(tag, create_tag=True, push_tag=True)
        if not tag_ok:
            print("⚠️  Warning: Tag setup may not be complete. Release may have incorrect URLs.")
            print()
    
    # Save notes to temporary file
    notes_file = '/tmp/release_notes.md'
    with open(notes_file, 'w') as f:
        f.write(notes)
    
    # Build command
    cmd = [
        'gh', 'release', 'create', tag,
        '--title', name,
        '--notes-file', notes_file,
        '--repo', repo,
    ]
    
    if draft:
        cmd.append('--draft')
    if prerelease:
        cmd.append('--prerelease')
    
    # Add files
    cmd.extend(files)
    
    print(f"Running: {' '.join(cmd[:10])}... (with {len(files)} files)")
    print("Please wait, uploading files...")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print()
        print(f"✅ Release '{name}' created successfully!")
        print(f"🔗 View at: https://github.com/{repo}/releases/tag/{tag}")
        print()
        print("✨ Images should now display inline in the release notes!")
        print(f"   Check: https://github.com/{repo}/releases/tag/{tag}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating release: {e}")
        print()
        print("Troubleshooting:")
        print(f"  1. Check the tag exists on GitHub: git ls-remote --tags origin {tag}")
        print(f"  2. Verify write access to {repo}")
        print(f"  3. Try: gh auth status")
        print()
        sys.exit(1)
    finally:
        # Clean up notes file
        if os.path.exists(notes_file):
            os.remove(notes_file)


def create_release_with_pygithub(
    tag: str,
    name: str,
    notes: str,
    files: List[str],
    repo: str = "uoa-eresearch/physicsnemo",
    draft: bool = False,
    prerelease: bool = False
) -> None:
    """Create release using PyGithub library."""
    
    try:
        from github import Github
    except ImportError:
        print("Error: PyGithub not installed. Install with: pip install PyGithub")
        print("Or use --use-gh-cli to use gh CLI instead")
        sys.exit(1)
    
    # Get token from environment
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set")
        print("Set it with: export GITHUB_TOKEN=your_token")
        sys.exit(1)
    
    print(f"Creating release on {repo}...")
    
    # Create GitHub client
    g = Github(token)
    repo_obj = g.get_repo(repo)
    
    # Create release
    release = repo_obj.create_git_release(
        tag=tag,
        name=name,
        message=notes,
        draft=draft,
        prerelease=prerelease
    )
    
    print(f"✅ Release '{name}' created!")
    
    # Upload assets
    for file_path in files:
        print(f"  Uploading {os.path.basename(file_path)}...")
        release.upload_asset(file_path)
    
    print(f"✅ All assets uploaded!")
    print(f"🔗 View at: {release.html_url}")


def main():
    parser = argparse.ArgumentParser(
        description='Create GitHub release with model checkpoints and plots'
    )
    parser.add_argument('--tag', required=True, help='Release tag (e.g., v1.0.0)')
    parser.add_argument('--name', required=True, help='Release name')
    parser.add_argument('--repo', default='uoa-eresearch/physicsnemo', help='GitHub repository')
    parser.add_argument('--draft', action='store_true', help='Create as draft')
    parser.add_argument('--prerelease', action='store_true', help='Mark as pre-release')
    parser.add_argument('--use-gh-cli', action='store_true', help='Use gh CLI instead of PyGithub')
    parser.add_argument('--max-plots', type=int, default=6, help='Max number of sample plots to include')
    parser.add_argument('--base-dir', default='.', help='Base directory (default: current)')
    parser.add_argument('--no-create-tag', action='store_true', help='Do not automatically create git tag')
    
    args = parser.parse_args()
    
    # Change to base directory
    os.chdir(args.base_dir)
    
    print("=" * 60)
    print("GitHub Release Creator for NZ GEFS-WHACS Models")
    print("=" * 60)
    print()
    
    # Find checkpoints
    print("📦 Finding checkpoints...")
    regression_ckpt_path = find_latest_checkpoint('checkpoints_regression')
    diffusion_ckpt_path = find_latest_checkpoint('checkpoints_diffusion')
    
    if not regression_ckpt_path:
        print("❌ Error: No regression checkpoint found in checkpoints_regression/")
        sys.exit(1)
    if not diffusion_ckpt_path:
        print("❌ Error: No diffusion checkpoint found in checkpoints_diffusion/")
        sys.exit(1)
    
    regression_info = CheckpointInfo(regression_ckpt_path, 'regression')
    diffusion_info = CheckpointInfo(diffusion_ckpt_path, 'diffusion')
    
    print(f"  ✓ Regression: {regression_info.filename} ({regression_info.size_mb:.1f} MB)")
    print(f"  ✓ Diffusion: {diffusion_info.filename} ({diffusion_info.size_mb:.1f} MB)")
    print()
    
    # Load configs
    print("📋 Loading configuration files...")
    regression_config = load_config_info('conf/config_train_gefs_WHACS_regression.yaml')
    diffusion_config = load_config_info('conf/config_train_gefs_WHACS.yaml')
    print(f"  ✓ Regression config: model_size={regression_config['model_size']}")
    print(f"  ✓ Diffusion config: model_size={diffusion_config['model_size']}")
    print()
    
    # Find sample plots
    print("🖼️  Finding sample plots...")
    plots = find_sample_plots('inference/plots', max_plots=args.max_plots)
    print(f"  ✓ Found {len(plots)} sample plots")
    for plot in plots:
        print(f"    - {os.path.basename(plot)}")
    print()
    
    # Generate release notes
    print("📝 Generating release notes...")
    notes = generate_release_notes(
        regression_info, diffusion_info,
        regression_config, diffusion_config,
        plots,
        tag=args.tag,
        repo=args.repo
    )
    print("  ✓ Release notes generated")
    print()
    
    # Prepare file list
    files = [regression_ckpt_path, diffusion_ckpt_path] + plots
    
    print(f"📤 Preparing to upload {len(files)} files:")
    print(f"  - 2 checkpoints")
    print(f"  - {len(plots)} plots")
    print()
    
    # Create release
    if args.use_gh_cli:
        create_release_with_gh_cli(
            args.tag, args.name, notes, files,
            repo=args.repo, draft=args.draft, prerelease=args.prerelease,
            create_tag=not args.no_create_tag
        )
    else:
        create_release_with_pygithub(
            args.tag, args.name, notes, files,
            repo=args.repo, draft=args.draft, prerelease=args.prerelease
        )
    
    print()
    print("=" * 60)
    print("✨ Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
