To generate slides:

    manim-slides render main.py LinearScene \
        && manim-slides convert LinearScene result.html --open

To preview locally quickly:

    manim -pql main.py SGD
