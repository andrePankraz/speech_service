import logging
from speech_service.nllb_manager import NllbManager
import sys

logger = logging.getLogger(__name__)


def test_translate():
    logger.debug("TEST")
    nllbManager = NllbManager()
    src_texts = ['Essential to the food chain\u200b', 'And there’s more to mosquitoes than pollination: if they weren’t around, our ecosystem would change entirely. When just one species disappears, it almost always has a knock-on effect. As with the rest of the animal kingdom, mosquitoes are an essential part of the food chain; they’re an important food source for many fish, reptiles and birds. \u200b', '', 'Indeed, they’re so crucial to many bird species in the Arctic tundra that these birds travel to mosquito-heavy regions every year to eat the insects that hatch there during summer. At this time of year, these regions have the highest concentration of mosquitoes on the planet.\u200b', '',
                 'Mosquitoes also make up a large part of the diet of certain fish species, especially the aptly named mosquito fish. These can consume thousands of mosquito larvae a day. And let’s not forget the bats, frogs, dragonflies, birds and other fish that rely on mosquitoes as food, too.\u200b', '', 'If mosquitoes were to disappear, the animals that eat them might stop living in or visiting certain areas. So, while we relax in our gardens, thinking about how we’d like to get rid of these annoyances once and for all, I’m afraid it’s not that simple. There’s much more to the tiny mosquito than itchy bites and ruined holidays – and, as it seems they’re here to stay, we’d better learn to live with them.\u200b']
    src_lang = nllbManager.identify_language(src_texts[0])
    tgt_texts = nllbManager.translate(src_texts, src_lang, 'deu_Latn')
    logger.info(
        f"Result from {src_lang=}: {tgt_texts}")
