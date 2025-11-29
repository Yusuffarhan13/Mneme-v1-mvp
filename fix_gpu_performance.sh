#!/bin/bash
# Fix RTX 4090 Laptop GPU Power/Performance Issues

echo "Fixing GPU Performance Settings..."
echo "======================================="

# Set persistence mode (keeps GPU ready)
sudo nvidia-smi -pm 1

# Set power limit to maximum (175W for RTX 4090 Laptop)
sudo nvidia-smi -pl 175

# Force performance mode (not available on all laptops)
sudo nvidia-smi --applications-clocks=DEFAULT || echo "Note: Application clocks not supported on this laptop"

# Lock clocks to max (alternative method)
sudo nvidia-smi -lgc 2100 || echo "Note: Clock locking not supported"

echo ""
echo "======================================="
echo "Settings applied! Check with:"
echo "  nvidia-smi -q -d POWER,PERFORMANCE,CLOCK"
echo ""
echo "IMPORTANT for LAPTOP GPUs:"
echo "- Make sure laptop is plugged into AC power"
echo "- Check BIOS/UEFI for GPU power settings"
echo "- Some laptops limit GPU power in battery mode"
echo "- Check your laptop's control center (MSI Dragon Center, etc.)"
echo "======================================="
