#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from docutils import nodes
from docutils.parsers.rst.roles import set_classes
import importlib

def list_registry_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    # Extract module and function name from the argument
    try:
        module_name, registry_name = text.rsplit('.', 1)
        module = importlib.import_module(module_name)
        registry = getattr(module, registry_name)

        # Execute the function and capture its output
        output_list = registry.list()

        # Format the output as a string with quotes and pipes
        formatted_output = '(' + ' | '.join(f'"{name}"' for name in output_list) + ')'

        # Create an inline node with the formatted output
        set_classes(options)
        node = nodes.inline(rawtext, formatted_output, **options)
        return [node], []
    except Exception as e:
        error = inliner.reporter.error(
            f"Error processing role 'list_registry_role': {str(e)}",
            nodes.literal_block(rawtext, rawtext), line=lineno)
        return [inliner.problematic(rawtext, rawtext, error)], [error]

def setup(app):
    app.add_role('list-registry', list_registry_role)