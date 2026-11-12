//! Позиция свидетельства (stance) и форма вопроса.
//!
//! **Почему модуль появился.** Спека `_gh/ARGUS-OS1/docs/ARGUS_Internal_Process_Idle_Inference_2026-11-12.md`
//! §3.4 требует для `Resolved` сдвиг `LR = 3.0` (подтверждение) либо `LR = 1/3`
//! (противоречие). Но проба (Europe PMC) отдаёт заголовок, абстракт и меру
//! релевантности — а **релевантность ≠ подтверждение**. В коде этот разрыв был закрыт
//! веткой `Outcome::Evidence`: идентификаторы пишутся, вероятность не двигается.
//! Наблюдение 13 ноя 2026: 15 циклов подряд `Resolved`, все 15 — `0.500 → 0.500`,
//! т.е. `prior_revision` (планка спеки §5: 0.2–1.0) = 0. Цикл тратил бюджет и API,
//! не меняя ни одного убеждения.
//!
//! **Что здесь.** Детерминированный классификатор с правом воздержаться:
//! - `Neutral` — состояние по умолчанию. Тогда исход остаётся `Evidence` и
//!   вероятность НЕ двигается (поведение M2 сохраняется);
//! - `Support` / `Contradict` — только при уверенности `>= ACT_CONF` и без
//!   конфликтующих сигналов;
//! - каждое решение пишется в цикл с источником и уверенностью — аудируемо.
//!
//! **Честное ограничение.** Это НЕ entailment-модель: лексические маркеры
//! (отрицания, утверждения) плюс якоря (стемы вопроса, числа с единицами) — грубый
//! прокси. Отсюда высокий порог действия, запрет действовать по одной релевантности
//! и обязательная запись воздержаний в отчёт (метрика `stance_abstain`).
//!
//! **Форма вопроса.** Приговор «p = 0.5» имеет смысл только для бинарной претензии.
//! Поисковая фраза («fiducial autofocus drift long-term time-lapse») — не претензия,
//! её вероятность не определена; количественный вопрос («насколько дрейфует фокус»)
//! требует оценки, а не подтверждения. Обе формы помечаются и к сдвигу p не допускаются.

/// Форма вопроса: к чему вообще применима байесовская бухгалтерия.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum QForm {
    /// бинарная претензия: «держит ли X при Y» — p осмысленна, LR применим
    Proposition,
    /// количественная оценка: «насколько велик X» — нужна величина, а не «да/нет»
    Quantitative,
    /// поисковая тема/фраза: p не определена
    Topic,
}

impl QForm {
    pub fn as_str(&self) -> &'static str {
        match self {
            QForm::Proposition => "proposition",
            QForm::Quantitative => "quantitative",
            QForm::Topic => "topic",
        }
    }
    /// Допускает ли форма сдвиг вероятности свидетельством.
    pub fn allows_lr(&self) -> bool {
        matches!(self, QForm::Proposition)
    }
}

/// Форма вопроса по его тексту. Никакой сети, только структура.
pub fn detect_form(text: &str) -> QForm {
    let low = text.to_lowercase();
    let t = low.trim();
    // «… ли …» — единственный надёжный маркер бинарной претензии в русском
    let has_li = t
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .any(|w| w == "ли");
    // английские да/нет-вопросы: вспомогательный глагол в начале
    let en_aux = [
        "is ", "are ", "does ", "do ", "can ", "could ", "will ", "would ", "should ", "has ",
        "have ", "was ", "were ",
    ];
    if has_li || en_aux.iter().any(|a| t.starts_with(a)) {
        return QForm::Proposition;
    }
    let quant_ru = [
        "насколько",
        "какова",
        "каков",
        "какая",
        "какое",
        "какие",
        "сколько",
        "какую",
        "насколько",
    ];
    let quant_en = [
        "what is",
        "how much",
        "how many",
        "how long",
        "how far",
        "which ",
        "what fraction",
    ];
    if quant_ru.iter().any(|a| t.starts_with(a)) || quant_en.iter().any(|a| t.starts_with(a)) {
        return QForm::Quantitative;
    }
    QForm::Topic
}

/// Позиция свидетельства относительно претензии.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stance {
    Support,
    Contradict,
    /// воздержание: сигналов нет, они слабые или конфликтуют
    Neutral,
}

impl Stance {
    pub fn as_str(&self) -> &'static str {
        match self {
            Stance::Support => "support",
            Stance::Contradict => "contradict",
            Stance::Neutral => "neutral",
        }
    }
}

/// Порог, ниже которого классификатор молчит. Одна лексическая зацепка не двигает веру.
pub const ACT_CONF: f32 = 0.70;
/// Минимальная доля стемов вопроса, найденных в свидетельстве (якорь темы).
pub const MIN_ANCHOR_RATIO: f32 = 0.35;
/// Минимум совпавших стемов вопроса.
pub const MIN_ANCHORS: usize = 2;

#[derive(Debug, Clone)]
pub struct Decision {
    pub stance: Stance,
    pub confidence: f32,
    /// откуда решение: `lexical:title+abstract` / `lexical:title`
    pub source: &'static str,
    /// человекочитаемая причина (идёт в цикл и в отчёт)
    pub reason: String,
}

impl Decision {
    pub fn neutral(source: &'static str, reason: impl Into<String>) -> Self {
        Decision {
            stance: Stance::Neutral,
            confidence: 0.0,
            source,
            reason: reason.into(),
        }
    }
    pub fn acts(&self) -> bool {
        self.stance != Stance::Neutral && self.confidence >= ACT_CONF
    }
}

const CONTRA_CUES: [&str; 26] = [
    "no evidence",
    "not significantly",
    "no significant",
    "does not",
    "did not",
    "was not",
    "were not",
    "fails to",
    "failed to",
    "failure to",
    "absence of",
    "unable to",
    "insufficient",
    "not detected",
    "not supported",
    "no association",
    "no difference",
    "no effect",
    "contradict",
    "refute",
    "in contrast",
    "не обнаружено",
    "не выявлено",
    "не подтвержда",
    "противореч",
    "не удалось",
];

const SUPPORT_CUES: [&str; 24] = [
    "we show",
    "we demonstrated",
    "we demonstrate",
    "we found that",
    "we report",
    "we confirm",
    "confirmed that",
    "evidence that",
    "consistent with",
    "in agreement",
    "supports the",
    "significantly higher",
    "significantly lower",
    "outperforms",
    "outperformed",
    "improves",
    "improved",
    "validated",
    "is achieved",
    "was achieved",
    "показано",
    "подтвержд",
    "согласуется",
    "достигнут",
];

fn stems(text: &str) -> Vec<String> {
    const STOP: [&str; 30] = [
        "the", "and", "for", "with", "from", "this", "that", "are", "was", "has", "its", "not",
        "can", "than", "ли", "или", "при", "для", "что", "как", "чем", "есть", "было", "были",
        "будет", "держит", "падает", "нужен", "нужна", "достаточно",
    ];
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 3)
        .filter(|w| !STOP.contains(w))
        .map(|w| w.chars().take(5).collect::<String>())
        .collect()
}

/// Числовые якоря вопроса: «1», «72», «0.5», «250» — с единицами не связываем,
/// важно само присутствие числа в свидетельстве.
fn numbers(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_digit() || (ch == '.' && !cur.is_empty()) {
            cur.push(ch);
        } else if !cur.is_empty() {
            if cur.trim_matches('.').len() >= 1 {
                out.push(cur.clone());
            }
            cur.clear();
        }
    }
    if !cur.is_empty() && cur.trim_matches('.').len() >= 1 {
        out.push(cur);
    }
    out.retain(|n| n != "." && n != "..");
    out.dedup();
    out
}

fn count_cues(hay: &str, cues: &[&str]) -> usize {
    cues.iter().filter(|c| hay.contains(*c)).count()
}

/// Классифицировать свидетельство относительно вопроса.
/// `abstract_text` пустой → работаем только по заголовку (уверенность ниже).
pub fn classify(question: &str, title: &str, abstract_text: &str) -> Decision {
    let has_abs = !abstract_text.trim().is_empty();
    let source: &'static str = if has_abs {
        "lexical:title+abstract"
    } else {
        "lexical:title"
    };
    let ev = format!("{} {}\n{}", title, abstract_text, "").to_lowercase();
    if ev.trim().is_empty() {
        return Decision::neutral(source, "пустое свидетельство");
    }

    // якорь темы: доля стемов вопроса, встречающихся в свидетельстве.
    // Если в вопросе есть латинские термы (а литература почти всегда английская),
    // знаменатель — только они: иначе русские слова вопроса разбавляют долю
    // и релевантное свидетельство выглядит «нерелевантным» (якорей 1/11).
    let mut q_stems = stems(question);
    let latin: Vec<String> = q_stems
        .iter()
        .filter(|s| s.chars().all(|c| c.is_ascii_alphanumeric()))
        .cloned()
        .collect();
    if latin.len() >= 2 {
        q_stems = latin;
    }
    if q_stems.is_empty() {
        return Decision::neutral(source, "в вопросе нет содержательных термов");
    }
    let hits = q_stems.iter().filter(|w| ev.contains(w.as_str())).count();
    let ratio = hits as f32 / q_stems.len() as f32;
    if hits < MIN_ANCHORS || ratio < MIN_ANCHOR_RATIO {
        return Decision::neutral(
            source,
            format!("тема не подтверждена лексически (якорей {hits}/{}, {ratio:.2})", q_stems.len()),
        );
    }

    // числовые якоря: совпадение числа — единственный «твёрдый» сигнал помимо слов
    let nums = numbers(question);
    let num_hit = nums.iter().filter(|n| ev.contains(n.as_str())).count();

    // Отрицания считаем по СЫРОМУ тексту (иначе «no evidence» само себя стирает),
    // а подтверждения — по нормализованному: «no evidence that» не должно засчитывать
    // подтверждающий маркер «evidence that» (поймано тестом 13 ноя).
    let contra = count_cues(&ev, &CONTRA_CUES);
    let ev_pos = ev
        .replace("no evidence that", "no_evidence:")
        .replace("not evidence that", "no_evidence:")
        .replace("no evidence of", "no_evidence:")
        .replace("no support", "no_support")
        .replace("does not support", "not_support")
        .replace("did not support", "not_support");
    let supp = count_cues(&ev_pos, &SUPPORT_CUES);

    if contra > 0 && supp > 0 {
        return Decision::neutral(
            source,
            format!("конфликт маркеров: подтверждающих {supp}, отрицающих {contra}"),
        );
    }
    if contra == 0 && supp == 0 {
        // тема совпала, позиции нет — самый частый и самый честный исход
        return Decision::neutral(
            source,
            format!("тема совпала (якорей {hits}, чисел {num_hit}), но позиции в тексте нет"),
        );
    }

    let strong = |cue_n: usize| -> f32 {
        let mut c = 0.50_f32;
        if num_hit > 0 {
            c += 0.15;
        }
        if ratio >= 0.60 {
            c += 0.10;
        }
        if cue_n >= 2 {
            c += 0.10;
        }
        if !has_abs {
            c -= 0.10; // только заголовок — слабее
        }
        c.clamp(0.0, 0.95)
    };

    let (stance, cues, conf) = if contra > 0 {
        (Stance::Contradict, contra, strong(contra))
    } else {
        (Stance::Support, supp, strong(supp))
    };
    if conf < ACT_CONF {
        // маркер есть, но свидетельство слабое: молчим, а не «слегка подтверждаем»
        return Decision::neutral(
            source,
            format!(
                "низкая уверенность {conf:.2} < {ACT_CONF} (якорей {hits}, чисел {num_hit}/{})",
                nums.len()
            ),
        );
    }
    Decision {
        stance,
        confidence: conf,
        source,
        reason: format!(
            "маркеров {cues}, чисел совпало {num_hit}/{}, якорей {hits}",
            nums.len()
        ),
    }
}

/// Свод по нескольким свидетельствам одного цикла.
/// Противоречие между свидетельствами → воздержание (литература спорит — не наш вывод).
pub fn aggregate(decisions: &[Decision]) -> Decision {
    let acts: Vec<&Decision> = decisions.iter().filter(|d| d.acts()).collect();
    if acts.is_empty() {
        return Decision::neutral(
            "lexical:aggregate",
            format!("нет уверенных позиций среди {} свидетельств", decisions.len()),
        );
    }
    let sup = acts.iter().find(|d| d.stance == Stance::Support);
    let con = acts.iter().find(|d| d.stance == Stance::Contradict);
    match (sup, con) {
        (Some(s), None) => (*s).clone(),
        (None, Some(c)) => (*c).clone(),
        (Some(_), Some(_)) => Decision::neutral(
            "lexical:aggregate",
            "свидетельства расходятся: есть и подтверждающие, и отрицающие",
        ),
        (None, None) => Decision::neutral("lexical:aggregate", "нет позиций"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_propositions_in_russian_and_english() {
        assert_eq!(
            detect_form("Держит ли fiducial-autofocus дрейф < 1 px за 72 ч на OpenFlexure?"),
            QForm::Proposition
        );
        assert_eq!(
            detect_form("Сходится ли Centrin1-GFP и SAS-4::mCherry на одном центриоле"),
            QForm::Proposition
        );
        assert_eq!(
            detect_form("Does fiducial autofocus hold drift below 1 px for 72 h?"),
            QForm::Proposition
        );
    }

    #[test]
    fn detects_quantitative_and_topic() {
        assert_eq!(
            detect_form("Насколько дрейфует фокус при многочасовой таймлапс-съёмке?"),
            QForm::Quantitative
        );
        assert_eq!(
            detect_form("Какая точность автофокуса по фидуциальным меткам достигнута?"),
            QForm::Quantitative
        );
        assert_eq!(
            detect_form("fiducial autofocus drift long-term time-lapse open microscope"),
            QForm::Topic
        );
        assert!(!QForm::Topic.allows_lr());
        assert!(QForm::Proposition.allows_lr());
    }

    #[test]
    fn abstains_when_topic_matches_but_no_stance_cue() {
        let d = classify(
            "Держит ли fiducial-autofocus дрейф < 1 px за 72 ч",
            "Fiducial markers for autofocus drift correction in long-term imaging",
            "We describe a marker-based autofocus for time-lapse microscopy of open hardware.",
        );
        assert_eq!(d.stance, Stance::Neutral, "{}", d.reason);
        assert!(!d.acts());
    }

    #[test]
    fn abstains_on_bare_relevance_even_with_cues() {
        // тема совпала слабо → позиция не засчитывается, даже если есть «we show»
        let d = classify(
            "Держит ли fiducial-autofocus дрейф < 1 px за 72 ч",
            "Fiducial autofocus",
            "We show that autofocus works.",
        );
        assert_eq!(d.stance, Stance::Neutral, "{}", d.reason);
    }

    #[test]
    fn contradiction_with_number_acts() {
        let d = classify(
            "Держит ли fiducial-autofocus дрейф < 1 px за 72 ч",
            "Fiducial autofocus drift in long-term time-lapse imaging",
            "We found no evidence that fiducial autofocus holds drift below 1 px over 72 h; \
             drift reached 7 px and the method does not support 72 h acquisition.",
        );
        assert_eq!(d.stance, Stance::Contradict, "{}", d.reason);
        assert!(d.acts(), "conf={}", d.confidence);
    }

    #[test]
    fn support_needs_anchor_ratio_and_number() {
        let d = classify(
            "Держит ли fiducial-autofocus дрейф < 1 px за 72 ч",
            "Fiducial autofocus holds drift below 1 px for 72 h in long-term time-lapse",
            "We demonstrated that fiducial autofocus holds drift below 1 px for 72 h; \
             measurements showed 0.6 px over 72 h of continuous acquisition.",
        );
        assert_eq!(d.stance, Stance::Support, "{}", d.reason);
        assert!(d.acts(), "conf={}", d.confidence);
    }

    #[test]
    fn conflicting_cues_abstain() {
        let d = classify(
            "Держит ли fiducial-autofocus дрейф < 1 px за 72 ч",
            "Fiducial autofocus drift long-term time-lapse imaging 1 px 72 h",
            "We show improved autofocus, yet we found no evidence of drift below 1 px over 72 h.",
        );
        assert_eq!(d.stance, Stance::Neutral, "{}", d.reason);
    }

    #[test]
    fn aggregate_prefers_single_confident_stance_and_flags_conflict() {
        let sup = Decision {
            stance: Stance::Support,
            confidence: 0.9,
            source: "test",
            reason: "s".into(),
        };
        let con = Decision {
            stance: Stance::Contradict,
            confidence: 0.9,
            source: "test",
            reason: "c".into(),
        };
        assert_eq!(aggregate(&[sup.clone()]).stance, Stance::Support);
        assert_eq!(aggregate(&[sup, con]).stance, Stance::Neutral);
    }

    #[test]
    fn latin_terms_dominate_anchors_for_russian_questions() {
        // русский вопрос + английский абстракт: знаменатель — латинские термы
        let d = classify(
            "Сходится ли Centrin1-GFP и SAS-4::mCherry на одном центриоле?",
            "Centrin1-GFP and SAS-4::mCherry localize to the same centriole",
            "We demonstrated that Centrin1-GFP and SAS-4::mCherry co-localize at the same \
             centriole in human fibroblasts; 92% of centrioles showed both markers.",
        );
        assert!(
            matches!(d.stance, Stance::Support | Stance::Neutral),
            "{:?} {}",
            d.stance,
            d.reason
        );
        assert!(!d.reason.contains("якорей 1/11"), "{}", d.reason);
    }

    #[test]
    fn numbers_extracted_from_question() {
        let n = numbers("дрейф < 1 px за 72 ч при 0.5");
        assert!(n.contains(&"1".to_string()));
        assert!(n.contains(&"72".to_string()));
        assert!(n.contains(&"0.5".to_string()));
    }
}
