# from whisper.tokenizer import LANGUAGES as WHISPER_LANGUAGES

'''
This file was created by ]init[ AG 2022.

Module for NLLB-Languages.
'''
LANGUAGES = {
    # 'Flores200 Language Id': ['Language Name', 'Whisper Language Id']
    # Flores see: https://github.com/facebookresearch/flores/tree/main/flores200
    # Whisper languages: see import above
    'ace_Arab': ['Acehnese (Arabic script)', None],
    'ace_Latn': ['Acehnese (Latin script)', None],
    'acm_Arab': ['Mesopotamian Arabic', 'ar'],
    'acq_Arab': ['Ta’izzi-Adeni Arabic', 'ar'],
    'aeb_Arab': ['Tunisian Arabic', 'ar'],
    'afr_Latn': ['Afrikaans', 'af'],
    'ajp_Arab': ['South Levantine Arabic', 'ar'],
    'aka_Latn': ['Akan', None],
    'amh_Ethi': ['Amharic', 'am'],
    'apc_Arab': ['North Levantine Arabic', 'ar'],
    'arb_Arab': ['Modern Standard Arabic', 'ar'],
    'arb_Latn': ['Modern Standard Arabic (Romanized)', None],
    'ars_Arab': ['Najdi Arabic', 'ar'],
    'ary_Arab': ['Moroccan Arabic', 'ar'],
    'arz_Arab': ['Egyptian Arabic', 'ar'],
    'asm_Beng': ['Assamese', 'as'],
    'ast_Latn': ['Asturian', None],
    'awa_Deva': ['Awadhi', None],
    'ayr_Latn': ['Central Aymara', None],
    'azb_Arab': ['South Azerbaijani', 'az'],
    'azj_Latn': ['North Azerbaijani', 'az'],
    'bak_Cyrl': ['Bashkir', 'ba'],
    'bam_Latn': ['Bambara', None],
    'ban_Latn': ['Balinese', None],
    'bel_Cyrl': ['Belarusian', 'be'],
    'bem_Latn': ['Bemba', None],
    'ben_Beng': ['Bengali', 'bn'],
    'bho_Deva': ['Bhojpuri', None],
    'bjn_Arab': ['Banjar (Arabic script)', None],
    'bjn_Latn': ['Banjar (Latin script)', None],
    'bod_Tibt': ['Standard Tibetan', 'bo'],
    'bos_Latn': ['Bosnian', 'bs'],
    'bug_Latn': ['Buginese', None],
    'bul_Cyrl': ['Bulgarian', 'bg'],
    'cat_Latn': ['Catalan', 'ca'],
    'ceb_Latn': ['Cebuano', None],
    'ces_Latn': ['Czech', 'cs'],
    'cjk_Latn': ['Chokwe', None],
    'ckb_Arab': ['Central Kurdish', None],
    'crh_Latn': ['Crimean Tatar', None],
    'cym_Latn': ['Welsh', 'cy'],
    'dan_Latn': ['Danish', 'da'],
    'deu_Latn': ['German', 'de'],
    'dik_Latn': ['Southwestern Dinka', None],
    'dyu_Latn': ['Dyula', None],
    'dzo_Tibt': ['Dzongkha', None],
    'ell_Grek': ['Greek', 'el'],
    'eng_Latn': ['English', 'en'],
    'epo_Latn': ['Esperanto', None],
    'est_Latn': ['Estonian', 'et'],
    'eus_Latn': ['Basque', 'eu'],
    'ewe_Latn': ['Ewe', None],
    'fao_Latn': ['Faroese', 'fo'],
    'fij_Latn': ['Fijian', None],
    'fin_Latn': ['Finnish', 'fi'],
    'fon_Latn': ['Fon', None],
    'fra_Latn': ['French', 'fr'],
    'fur_Latn': ['Friulian', None],
    'fuv_Latn': ['Nigerian Fulfulde', None],
    'gla_Latn': ['Scottish Gaelic', None],
    'gle_Latn': ['Irish', None],
    'glg_Latn': ['Galician', 'gl'],
    'grn_Latn': ['Guarani', None],
    'guj_Gujr': ['Gujarati', 'gu'],
    'hat_Latn': ['Haitian Creole', 'ht'],
    'hau_Latn': ['Hausa', 'ha'],
    'heb_Hebr': ['Hebrew', 'iw'],
    'hin_Deva': ['Hindi', 'hi'],
    'hne_Deva': ['Chhattisgarhi', None],
    'hrv_Latn': ['Croatian', 'hr'],
    'hun_Latn': ['Hungarian', 'hu'],
    'hye_Armn': ['Armenian', 'hy'],
    'ibo_Latn': ['Igbo', None],
    'ilo_Latn': ['Ilocano', None],
    'ind_Latn': ['Indonesian', 'id'],
    'isl_Latn': ['Icelandic', 'is'],
    'ita_Latn': ['Italian', 'it'],
    'jav_Latn': ['Javanese', 'jw'],
    'jpn_Jpan': ['Japanese', 'ja'],
    'kab_Latn': ['Kabyle', None],
    'kac_Latn': ['Jingpho', None],
    'kam_Latn': ['Kamba', None],
    'kan_Knda': ['Kannada', 'kn'],
    'kas_Arab': ['Kashmiri (Arabic script)', None],
    'kas_Deva': ['Kashmiri (Devanagari script)', None],
    'kat_Geor': ['Georgian', 'ka'],
    'knc_Arab': ['Central Kanuri (Arabic script)', None],
    'knc_Latn': ['Central Kanuri (Latin script)', None],
    'kaz_Cyrl': ['Kazakh', 'kk'],
    'kbp_Latn': ['Kabiyè', None],
    'kea_Latn': ['Kabuverdianu', None],
    'khm_Khmr': ['Khmer', 'km'],
    'kik_Latn': ['Kikuyu', None],
    'kin_Latn': ['Kinyarwanda', None],
    'kir_Cyrl': ['Kyrgyz', None],
    'kmb_Latn': ['Kimbundu', None],
    'kmr_Latn': ['Northern Kurdish', None],
    'kon_Latn': ['Kikongo', None],
    'kor_Hang': ['Korean', 'ko'],
    'lao_Laoo': ['Lao', 'lo'],
    'lij_Latn': ['Ligurian', None],
    'lim_Latn': ['Limburgish', None],
    'lin_Latn': ['Lingala', 'ln'],
    'lit_Latn': ['Lithuanian', 'lt'],
    'lmo_Latn': ['Lombard', None],
    'ltg_Latn': ['Latgalian', None],
    'ltz_Latn': ['Luxembourgish', 'lb'],
    'lua_Latn': ['Luba-Kasai', None],
    'lug_Latn': ['Ganda', None],
    'luo_Latn': ['Luo', None],
    'lus_Latn': ['Mizo', None],
    'lvs_Latn': ['Standard Latvian', 'lv'],
    'mag_Deva': ['Magahi', None],
    'mai_Deva': ['Maithili', None],
    'mal_Mlym': ['Malayalam', 'ml'],
    'mar_Deva': ['Marathi', 'mr'],
    'min_Arab': ['Minangkabau (Arabic script)', None],
    'min_Latn': ['Minangkabau (Latin script)', None],
    'mkd_Cyrl': ['Macedonian', 'mk'],
    'plt_Latn': ['Plateau Malagasy', 'mg'],
    'mlt_Latn': ['Maltese', 'mt'],
    'mni_Beng': ['Meitei (Bengali script)', None],
    'khk_Cyrl': ['Halh Mongolian', 'mn'],
    'mos_Latn': ['Mossi', None],
    'mri_Latn': ['Maori', 'mi'],
    'mya_Mymr': ['Burmese', 'my'],
    'nld_Latn': ['Dutch', 'nl'],
    'nno_Latn': ['Norwegian Nynorsk', 'nn'],
    'nob_Latn': ['Norwegian Bokmål', 'no'],
    'npi_Deva': ['Nepali', 'ne'],
    'nso_Latn': ['Northern Sotho', None],
    'nus_Latn': ['Nuer', None],
    'nya_Latn': ['Nyanja', None],
    'oci_Latn': ['Occitan', 'oc'],
    'gaz_Latn': ['West Central Oromo', None],
    'ory_Orya': ['Odia', None],
    'pag_Latn': ['Pangasinan', None],
    'pan_Guru': ['Eastern Panjabi', 'pa'],
    'pap_Latn': ['Papiamento', None],
    'pes_Arab': ['Western Persian', 'fa'],
    'pol_Latn': ['Polish', 'pl'],
    'por_Latn': ['Portuguese', 'pt'],
    'prs_Arab': ['Dari', None],
    'pbt_Arab': ['Southern Pashto', 'ps'],
    'quy_Latn': ['Ayacucho Quechua', None],
    'ron_Latn': ['Romanian', 'ro'],
    'run_Latn': ['Rundi', None],
    'rus_Cyrl': ['Russian', 'ru'],
    'sag_Latn': ['Sango', None],
    'san_Deva': ['Sanskrit', 'sa'],
    'sat_Olck': ['Santali', None],
    'scn_Latn': ['Sicilian', None],
    'shn_Mymr': ['Shan', None],
    'sin_Sinh': ['Sinhala', 'si'],
    'slk_Latn': ['Slovak', 'sk'],
    'slv_Latn': ['Slovenian', 'sl'],
    'smo_Latn': ['Samoan', None],
    'sna_Latn': ['Shona', 'sn'],
    'snd_Arab': ['Sindhi', 'sd'],
    'som_Latn': ['Somali', 'so'],
    'sot_Latn': ['Southern Sotho', None],
    'spa_Latn': ['Spanish', 'es'],
    'als_Latn': ['Tosk Albanian', 'sq'],
    'srd_Latn': ['Sardinian', None],
    'srp_Cyrl': ['Serbian', 'sr'],
    'ssw_Latn': ['Swati', None],
    'sun_Latn': ['Sundanese', 'su'],
    'swe_Latn': ['Swedish', 'sv'],
    'swh_Latn': ['Swahili', 'sw'],
    'szl_Latn': ['Silesian', None],
    'tam_Taml': ['Tamil', 'ta'],
    'tat_Cyrl': ['Tatar', 'tt'],
    'tel_Telu': ['Telugu', 'te'],
    'tgk_Cyrl': ['Tajik', 'tg'],
    'tgl_Latn': ['Tagalog', 'tl'],
    'tha_Thai': ['Thai', 'th'],
    'tir_Ethi': ['Tigrinya', None],
    'taq_Latn': ['Tamasheq (Latin script)', None],
    'taq_Tfng': ['Tamasheq (Tifinagh script)', None],
    'tpi_Latn': ['Tok Pisin', None],
    'tsn_Latn': ['Tswana', None],
    'tso_Latn': ['Tsonga', None],
    'tuk_Latn': ['Turkmen', 'tk'],
    'tum_Latn': ['Tumbuka', None],
    'tur_Latn': ['Turkish', 'tr'],
    'twi_Latn': ['Twi', None],
    'tzm_Tfng': ['Central Atlas Tamazight', None],
    'uig_Arab': ['Uyghur', None],
    'ukr_Cyrl': ['Ukrainian', 'uk'],
    'umb_Latn': ['Umbundu', None],
    'urd_Arab': ['Urdu', 'ur'],
    'uzn_Latn': ['Northern Uzbek', 'uz'],
    'vec_Latn': ['Venetian', None],
    'vie_Latn': ['Vietnamese', 'vi'],
    'war_Latn': ['Waray', None],
    'wol_Latn': ['Wolof', None],
    'xho_Latn': ['Xhosa', None],
    'ydd_Hebr': ['Eastern Yiddish', 'yi'],
    'yor_Latn': ['Yoruba', 'yo'],
    'yue_Hant': ['Yue Chinese', 'zh'],
    'zho_Hans': ['Chinese (Simplified)', 'zh'],
    'zho_Hant': ['Chinese (Traditional)', 'zh'],
    'zsm_Latn': ['Standard Malay', 'ms'],
    'zul_Latn': ['Zulu', None]
}
