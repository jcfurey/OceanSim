#!/usr/bin/env bash
# Launch the built-in Deep Trekker REVOLUTION platform. In auto mode, a mounted
# OceanSim asset pack supplies the scanned MHL environment and detailed vehicle;
# otherwise bundled PBR environments and local CAD (or a generated URDF) are used.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCEANSIM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ASSET_ROOT="${OCEANSIM_ASSETS:-$OCEANSIM_ROOT/demo/assets}"
ENVIRONMENT="${OCEANSIM_ENVIRONMENT:-auto}"
RUNNER_ARGS=()

while (($#)); do
    case "$1" in
        --environment)
            if (($# < 2)); then
                echo "ERROR: --environment requires auto, builtin, reef, mhl, or a USD path" >&2
                exit 2
            fi
            ENVIRONMENT="$2"
            shift 2
            ;;
        --environment=*)
            ENVIRONMENT="${1#*=}"
            shift
            ;;
        *)
            RUNNER_ARGS+=("$1")
            shift
            ;;
    esac
done

BUILTIN_SCENE="$OCEANSIM_ROOT/demo/assets/environments/inspection_site.usdc"
if [[ ! -f "$BUILTIN_SCENE" ]]; then
    BUILTIN_SCENE="$OCEANSIM_ROOT/demo/revolution_scene.usda"
fi
MHL_SCENE="$ASSET_ROOT/collected_MHL/mhl_scaled.usd"
MHL_ROCK="$ASSET_ROOT/collected_rock/rock.usd"
SCENE_ARGS=()

case "$ENVIRONMENT" in
    auto)
        if [[ -f "$MHL_SCENE" && -f "$MHL_ROCK" ]]; then
            echo "[run_deeptrekker_revolution] environment: MHL asset-pack scene"
            # No --scene-usd: the runner's default environment adds MHL, its
            # collider/sonar reflectivity, and the inspection rock.
        else
            echo "[run_deeptrekker_revolution] environment: bundled subsea inspection site"
            SCENE_ARGS=(--scene-usd "$BUILTIN_SCENE")
        fi
        ;;
    builtin)
        echo "[run_deeptrekker_revolution] environment: bundled subsea inspection site"
        SCENE_ARGS=(--scene-usd "$BUILTIN_SCENE")
        ;;
    reef)
        echo "[run_deeptrekker_revolution] environment: rocky reef survey site"
        SCENE_ARGS=(--scene-usd "$OCEANSIM_ROOT/demo/assets/environments/rocky_reef.usdc")
        ;;
    mhl)
        if [[ ! -f "$MHL_SCENE" || ! -f "$MHL_ROCK" ]]; then
            echo "ERROR: MHL environment not found under $ASSET_ROOT" >&2
            echo "Set OCEANSIM_ASSETS on the host to the downloaded OceanSim_assets directory." >&2
            exit 2
        fi
        echo "[run_deeptrekker_revolution] environment: MHL asset-pack scene"
        ;;
    *)
        if [[ ! -f "$ENVIRONMENT" ]]; then
            echo "ERROR: environment USD not found: $ENVIRONMENT" >&2
            exit 2
        fi
        echo "[run_deeptrekker_revolution] environment: $ENVIRONMENT"
        SCENE_ARGS=(--scene-usd "$ENVIRONMENT")
        ;;
esac

exec "$SCRIPT_DIR/run_oceansim_ros2.sh" \
    --config "$OCEANSIM_ROOT/demo/revolution.json" \
    --asset-path "$ASSET_ROOT" \
    "${SCENE_ARGS[@]}" \
    --platform deeptrekker_revolution \
    --publish-static-tf \
    "${RUNNER_ARGS[@]}"
