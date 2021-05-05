#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Collection of tools for text-related modifications. '''

# =============================================================================
# Imports
# =============================================================================
import matplotlib.pyplot as plt
import textwrap

# =============================================================================
# Tools
# =============================================================================

def secure_str(string):
    return string.lower().replace(' ', '_')

def boldify_str(string):
    if plt.rcParams['text.usetex']:
        return ''.join([r'\textbf{',string,'}'])
    else:
        return string

def wrap_str(string):
    return textwrap.indent(textwrap.fill(string, width=80),'# ')
