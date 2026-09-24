/**
 * Answer languages, and the few interface labels that follow the chosen one.
 *
 * The answer itself is written by the backend in the chosen language. Source
 * extracts are never translated, and offline answers translate only their
 * headings (backend config.OFFLINE_LABELS). The labels here are the badges an
 * operator scans first, so they follow the language too; everything else in
 * the interface stays in copy.ts.
 */

export interface LanguageLabels {
  offline: string
  dataOnly: string
  copilot: string
}

export interface Language {
  code: string
  english: string
  native: string
  labels: LanguageLabels
}

export const languages: Language[] = [
  {
    code: 'en', english: 'English', native: 'English',
    labels: {
      offline: 'Offline — sources only, no generated answer',
      dataOnly: 'Offline — survey data and sources only, no generated answer',
      copilot: 'Mission Copilot',
    },
  },
  {
    code: 'hi', english: 'Hindi', native: 'हिन्दी',
    labels: {
      offline: 'ऑफ़लाइन — केवल स्रोत, कोई जनरेट किया गया उत्तर नहीं',
      dataOnly: 'ऑफ़लाइन — केवल सर्वे डेटा और स्रोत, कोई जनरेट किया गया उत्तर नहीं',
      copilot: 'मिशन कोपायलट',
    },
  },
  {
    code: 'ta', english: 'Tamil', native: 'தமிழ்',
    labels: {
      offline: 'ஆஃப்லைன் — மூலங்கள் மட்டும், உருவாக்கப்பட்ட பதில் இல்லை',
      dataOnly: 'ஆஃப்லைன் — ஆய்வுத் தரவும் மூலங்களும் மட்டும், உருவாக்கப்பட்ட பதில் இல்லை',
      copilot: 'மிஷன் கோபைலட்',
    },
  },
  {
    code: 'ml', english: 'Malayalam', native: 'മലയാളം',
    labels: {
      offline: 'ഓഫ്‌ലൈൻ — ഉറവിടങ്ങൾ മാത്രം, സൃഷ്ടിച്ച ഉത്തരമില്ല',
      dataOnly: 'ഓഫ്‌ലൈൻ — സർവേ ഡാറ്റയും ഉറവിടങ്ങളും മാത്രം, സൃഷ്ടിച്ച ഉത്തരമില്ല',
      copilot: 'മിഷൻ കോപൈലറ്റ്',
    },
  },
  {
    code: 'or', english: 'Odia', native: 'ଓଡ଼ିଆ',
    labels: {
      offline: 'ଅଫଲାଇନ୍ — କେବଳ ଉତ୍ସ, କୌଣସି ସୃଷ୍ଟ ଉତ୍ତର ନାହିଁ',
      dataOnly: 'ଅଫଲାଇନ୍ — କେବଳ ସର୍ଭେ ତଥ୍ୟ ଓ ଉତ୍ସ, କୌଣସି ସୃଷ୍ଟ ଉତ୍ତର ନାହିଁ',
      copilot: 'ମିଶନ୍ କୋପାଇଲଟ୍',
    },
  },
  {
    code: 'te', english: 'Telugu', native: 'తెలుగు',
    labels: {
      offline: 'ఆఫ్‌లైన్ — మూలాలు మాత్రమే, రూపొందించిన సమాధానం లేదు',
      dataOnly: 'ఆఫ్‌లైన్ — సర్వే డేటా మరియు మూలాలు మాత్రమే, రూపొందించిన సమాధానం లేదు',
      copilot: 'మిషన్ కోపైలట్',
    },
  },
  {
    code: 'bn', english: 'Bengali', native: 'বাংলা',
    labels: {
      offline: 'অফলাইন — শুধু উৎস, কোনো তৈরি উত্তর নেই',
      dataOnly: 'অফলাইন — শুধু জরিপের তথ্য ও উৎস, কোনো তৈরি উত্তর নেই',
      copilot: 'মিশন কোপাইলট',
    },
  },
  {
    code: 'kn', english: 'Kannada', native: 'ಕನ್ನಡ',
    labels: {
      offline: 'ಆಫ್‌ಲೈನ್ — ಮೂಲಗಳು ಮಾತ್ರ, ರಚಿಸಿದ ಉತ್ತರವಿಲ್ಲ',
      dataOnly: 'ಆಫ್‌ಲೈನ್ — ಸಮೀಕ್ಷೆ ದತ್ತಾಂಶ ಮತ್ತು ಮೂಲಗಳು ಮಾತ್ರ, ರಚಿಸಿದ ಉತ್ತರವಿಲ್ಲ',
      copilot: 'ಮಿಷನ್ ಕೋಪೈಲಟ್',
    },
  },
  {
    code: 'mr', english: 'Marathi', native: 'मराठी',
    labels: {
      offline: 'ऑफलाइन — फक्त स्रोत, तयार केलेले उत्तर नाही',
      dataOnly: 'ऑफलाइन — फक्त सर्वेक्षण डेटा आणि स्रोत, तयार केलेले उत्तर नाही',
      copilot: 'मिशन कोपायलट',
    },
  },
  {
    code: 'gu', english: 'Gujarati', native: 'ગુજરાતી',
    labels: {
      offline: 'ઑફલાઇન — માત્ર સ્રોતો, કોઈ બનાવેલો જવાબ નહીં',
      dataOnly: 'ઑફલાઇન — માત્ર સર્વે ડેટા અને સ્રોતો, કોઈ બનાવેલો જવાબ નહીં',
      copilot: 'મિશન કોપાયલટ',
    },
  },
]

export const defaultLanguage = 'en'

export function languageFor(code: string | null | undefined): Language {
  return languages.find((l) => l.code === code) ?? languages[0]
}
