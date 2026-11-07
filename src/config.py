from __future__ import annotations
# url based features
url_features: list[str] = [
    'URLLength',
    'DomainLength',
    'IsDomainIP',
    'TLDLength',
    'NoOfSubDomain',
    'HasObfuscation',
    'NoOfObfuscatedChar',
    'ObfuscationRatio',
    'NoOfLettersInURL',
    'LetterRatioInURL',
    'NoOfDegitsInURL',
    'DegitRatioInURL',
    'NoOfEqualsInURL',
    'NoOfQMarkInURL',
    'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL',
    'IsHTTPS',
    'CharContinuationRate',
    'TLD'
]

# lightweight webpage features
webpage_features = [
    'HasTitle',
    'HasDescription',
    'HasFavicon',
    'NoOfiFrame',
    'HasPasswordField',
    'HasHiddenFields',
    'HasSubmitButton',
    'HasExternalFormSubmit',
    'NoOfURLRedirect',
    'NoOfExternalRef',
    'NoOfSelfRef',
    'NoOfCSS',
    'NoOfJS',
    'NoOfImage',
    'DomainTitleMatchScore',
    'URLTitleMatchScore'
]

# features excluded for now
exclude_features = [
    'FILENAME',
    'URL',
    'Domain',
    'URLSimilarityIndex',
    'TLDLegitimateProb',
    'URLCharProb',
    'LineOfCode',
    'LargestLineLength',
    'Title',
    'Robots',
    'IsResponsive',
    'NoOfSelfRedirect',
    'NoOfPopup',
    'HasSocialNet',
    'HasCopyrightInfo',
    'Bank',
    'Pay',
    'Crypto',
]