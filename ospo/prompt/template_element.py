
def get_object_prompt():
    system = """You are a helpful assistant that generates common object spanning various categories, including animals, plants, fruits, household items, clothing, vehicles, food, musical instruments, and electronic devices."""

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate common objects spanning various categories.",
        },
        {"role": "<|Assistant|>", "content": 'dog, rose, apple, chair, shirt, car, pizza, guitar, cell phone'},
        {
            "role": "<|User|>",
            "content": "Generate common objects spanning various categories.",
        },
        {"role": "<|Assistant|>", "content": 'cat, cactus, banana, sofa, jacket, bicycle, sushi, piano, laptop'},
        {
            "role": "<|User|>",
            "content": f"Generate common objects spanning various categories.",
        },
        {"role": "<|Assistant|>", "content": ''},
    ]

    return system, conversation


def get_color_prompt():
    system = """You are a helpful assistant that generates common colors spanning various categories, including animals, plants, fruits, household items, clothing, vehicles, food, musical instruments, and electronic devices."""

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate common colors spanning various categories.\nOutput only simple color names (e.g., red, blue, gray). Avoid compound colors (e.g., dark gray, light blue).",
        },
        {"role": "<|Assistant|>", "content": 'red, blue, green, yellow, black, white, orange, pink, purple, brown'},
        {
            "role": "<|User|>",
            "content": "Generate common colors spanning various categories.\nOutput only simple color names (e.g., red, blue, gray). Avoid compound colors (e.g., dark gray, light blue).",
        },
        {"role": "<|Assistant|>", "content": 'cyan, magenta, lime, indigo, teal, navy, beige, maroon, olive, gold'},
        {
            "role": "<|User|>",
            "content": "Generate common colors spanning various categories.\nOutput only simple color names (e.g., red, blue, gray). Avoid compound colors (e.g., dark gray, light blue).",
        },
        {"role": "<|Assistant|>", "content": ''},
    ]

    return system, conversation


def get_texture_prompt():
    system = """You are a helpful assistant that generates common texture spanning various categories, including animals, plants, fruits, household items, clothing, vehicles, food, musical instruments, and electronic devices."""

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate common texture spanning various categories.\nAvoid compound texture.",
        },
        {"role": "<|Assistant|>", "content": 'rubber, metallic, leather, fabric, wooden, rough, smooth, soft, fluffy, glass'},
        {
            "role": "<|User|>",
            "content": "Generate common texture spanning various categories.\nAvoid compound texture.",
        },
        {"role": "<|Assistant|>", "content": 'gritty, silky, woolly, grainy, velvety, bumpy, slick, crinkled, coarse, porous'},
        {
            "role": "<|User|>",
            "content": "Generate common texture spanning various categories.\nAvoid compound texture.",
        },
        {"role": "<|Assistant|>", "content": ''},
    ]

    return system, conversation



def get_shape_prompt():
    system = """You are a helpful assistant that generates common shape spanning various categories, including animals, plants, fruits, household items, clothing, vehicles, food, musical instruments, and electronic devices.
Avoid containing objects names in the output."""

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate common shape spanning various categories.\nAvoid compound shape.\nAvoid object names like car, house...",
        },
        {"role": "<|Assistant|>", "content": 'diamond, square, pyramidal, triangular, rectangular, oval, short, teardrop, cubic, oblong'},
        {
            "role": "<|User|>",
            "content": "Generate common shape spanning various categories.\nAvoid compound shape.\nAvoid object names like car, house...",
        },
        {"role": "<|Assistant|>", "content": 'circular, small, spherical, conical, cylindrical, heart, big, spiral, tall'},
        {
            "role": "<|User|>",
            "content": "Generate common shape spanning various categories.\nAvoid compound shape.\nAvoid object names like car, house...",
        },
        {"role": "<|Assistant|>", "content": ''},
    ]

    return system, conversation



def get_spatial_prompt():
    system = """You are a helpful assistant that generates common spatial relative word or phrase spanning various categories, including animals, plants, fruits, household items, clothing, vehicles, food, musical instruments, and electronic devices.
Contain only one spatial relative phrase.
Avoid compound spatial relative word such as farther up for the right of, closer down for the left of, etc.
Avoid containing objects names such as image, window, screen, etc. in the output."""

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate common spatial relative word or phrase spanning various categories.\nAvoid compound spatial relative word.",
        },
        {"role": "<|Assistant|>", "content": 'in the left of, among, above, below, beside, opposite to, next to, above of, below of, beside of'},
        {
            "role": "<|User|>",
            "content": "Generate common spatial relative word or phrase spanning various categories.\nAvoid compound spatial relative word.",
        },
        {"role": "<|Assistant|>", "content": 'in the right of, in the middle of, in front of, hidden by, top of, next to'},
        {
            "role": "<|User|>",
            "content": "Generate common spatial relative word or phrase spanning various categories.\nAvoid compound spatial relative word.",
        },
        {"role": "<|Assistant|>", "content": ''},
    ]

    return system, conversation



def get_non_spatial_prompt():
    system = """You are an assistant dedicated to generating natural prompts that contain subjects and objects by using nonspatial relationship words such as wear, watch, speak, hold, have, run, look at, talk to, jump, play, walk with, stand on, and sit on."""

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using non-spatial relationship words.",
        },
        {"role": "<|Assistant|>", "content": 'Two friends are watching a movie together on a large TV screen.'},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using non-spatial relationship words.",
        },
        {"role": "<|Assistant|>", "content": 'Two athletes are running along the beach as the sun sets behind them.'},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using non-spatial relationship words.",
        },
        {"role": "<|Assistant|>", "content": ''},
    ]

    return system, conversation


def get_complex_prompt():
    system = """You are an assistant dedicated to generating natural compositional phrases or prompts, containing multiple objects (number ≥ 2) with one or several adjectives from color, shape, and texture descriptions and spatial (left/right/top/bottom/next to/near/on side of) or nonspatial relationships."""

    conversation = [
        {
            "role": "<|User|>",
            "content": "Please generate a compositional phrase or sentence containing multiple objects with one or several adjectives and relationships.",
        },
        {"role": "<|Assistant|>", "content": 'The fluffy white cat sat next to the black leather couch.'},
        {
            "role": "<|User|>",
            "content": "Please generate a compositional phrase or sentence containing multiple objects with one or several adjectives and relationships.",
        },
        {"role": "<|Assistant|>", "content": 'The sleek black phone rested beside the textured brown leather wallet.'},
        {
            "role": "<|User|>",
            "content": "Please generate a compositional phrase or sentence containing multiple objects with one or several adjectives and relationships.",
        },
        {"role": "<|Assistant|>", "content": 'The red spherical balloon floated above the striped rectangular kite and the green triangular flag.'},
        {
            "role": "<|User|>",
            "content": "Please generate a compositional phrase or sentence containing multiple objects with one or several adjectives and relationships.",
        },
        {"role": "<|Assistant|>", "content": 'The golden, sunlit leaves floated softly above the jagged, rust-colored rocks, their delicate, lacy shapes casting playful shadows on the uneven ground.'},
        {
            "role": "<|User|>",
            "content": "Please generate a compositional phrase or sentence containing multiple objects with one or several adjectives and relationships.",
        },
        {"role": "<|Assistant|>", "content": ''},
    ]

    return system, conversation


def get_prompt_element(category, processor):

    if category == 'object':
        system, conversation = get_object_prompt()
    elif category == 'color':
        system, conversation = get_color_prompt()
    elif category == 'texture':
        system, conversation = get_texture_prompt()
    elif category == 'shape':
        system, conversation = get_shape_prompt()
    elif category == 'spatial':
        system, conversation = get_spatial_prompt()

    # They are not for generating object items, but for generating prompts.
    elif category == 'non-spatial':
        system, conversation = get_non_spatial_prompt()
    elif category == 'complex':
        system, conversation = get_complex_prompt()

    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt=system,
    )

    return sft_format
