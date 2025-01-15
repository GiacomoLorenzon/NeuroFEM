#!/usr/bin/env python3
#
################################################################################
# NeuroFEM project - 2023
#
# Generate colormap xml file for paraview.
#
# ------------------------------------------------------------------------------
# Author: Giacomo Lorenzon <giacomo.lorenzon@mail.polimi.it>.
################################################################################


def generate_xml_color_map():
    xml_content = """<ColorMaps>
    <ColorMap space="CIELAB" indexedLookup="false" name="CustomColormap">"""

    colors = [
        (0.000000, 1, 140, 27, 16),
        (0.100000, 1, 246, 200, 68),
        (0.200000, 1, 82, 182, 152),
        (0.300000, 1, 32, 77, 136),
        (0.400000, 1, 145, 145, 145),
    ]

    for i in range(2056):
        color_index = i % len(colors)
        point = colors[color_index]
        xml_content += f"""
        <!-- Define color transitions at integer value {i} -->
        <Point x="{i}" o="{point[1]}" r="{point[2]}" g="{point[3]}" b="{point[4]}"/>"""

    xml_content += """
        <!-- Define NaN color -->
        <NaN r="0.25" g="0" b="0"/>
    </ColorMap>
</ColorMaps>"""

    return xml_content


def save_xml_file(xml_content, filename):
    with open(filename, "w") as file:
        file.write(xml_content)


# Generate XML color map content
xml_color_map = generate_xml_color_map()

# Save XML content to a file
filename = "custom_colormap.xml"
save_xml_file(xml_color_map, filename)
print(f"XML colormap saved to '{filename}'")
