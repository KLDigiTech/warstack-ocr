from PIL import Image, ImageDraw

img = Image.open("IMG_1437.jpg")
w, h = img.size
print(f"Image size: {w}x{h}")
draw = ImageDraw.Draw(img)

zones = {
    # OK
    "placement":    (0.420, 0.095, 0.580, 0.200),
    "team_kills":   (0.355, 0.330, 0.430, 0.395),

    # OK
    "p1_pseudo":    (0.128, 0.595, 0.243, 0.618),
    "p2_pseudo":    (0.333, 0.595, 0.448, 0.618),
    "p3_pseudo":    (0.538, 0.595, 0.653, 0.618),
    "p4_pseudo":    (0.743, 0.595, 0.858, 0.618),

# BLEU - Kills
    "p1_kills":     (0.163, 0.792, 0.191, 0.819),
    "p2_kills":     (0.368, 0.792, 0.396, 0.819),
    "p3_kills":     (0.572, 0.792, 0.600, 0.819),
    "p4_kills":     (0.775, 0.792, 0.803, 0.819),

    # VIOLET - Deaths (+0.002 droite)
    "p1_deaths":    (0.195, 0.792, 0.223, 0.819),
    "p2_deaths":    (0.400, 0.792, 0.428, 0.819),
    "p3_deaths":    (0.605, 0.792, 0.633, 0.819),
    "p4_deaths":    (0.810, 0.792, 0.838, 0.819),
}

colors = {
    "placement": "red",
    "team_kills": "lime",
    "pseudo": "yellow",
    "kills": "cyan",
    "deaths": "violet",
}

for name, (x1, y1, x2, y2) in zones.items():
    color = "white"
    if "placement" in name: color = "red"
    elif "team_kills" in name: color = "lime"
    elif "pseudo" in name: color = "yellow"
    elif "kills" in name: color = "cyan"
    elif "deaths" in name: color = "violet"

    draw.rectangle(
        [int(w*x1), int(h*y1), int(w*x2), int(h*y2)],
        outline=color, width=3
    )
    draw.text((int(w*x1), int(h*y1) - 20), name, fill=color)

img.save("debug_zones.png")
print("Sauvegardé : debug_zones.png")