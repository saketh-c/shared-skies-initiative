import { useContext } from "react";
import { LanguageContext } from "../App";
import { t } from "../i18n";
import { pm25Color } from "../utils/aqi";

// Swatch colors are sampled from the choropleth gradient (pm25Color) so the
// guide matches the map and legend exactly. Each band uses a representative
// PM2.5 value: Good/Moderate/Elevated at the band's upper edge, High at a
// mid-band red. (Previously these were hardcoded greens/yellows/red that the
// gradient never emits — #10b981 / #eab308 / #b30000.)
const COLOR_MAP = {
  good:     pm25Color(6),   // #00b894
  moderate: pm25Color(9),   // #FFD700
  elevated: pm25Color(15),  // #E8590C
  high:     pm25Color(35),  // #c03636 (mid-High; gradient runs #FF6B6B→#800000)
};

export default function AirQualityGuide() {
  const { lang } = useContext(LanguageContext);

  const levels = ["good", "moderate", "elevated", "high"].map((k) => ({
    key: k,
    name: t(lang, `guide.levels.${k}.name`),
    range: t(lang, `guide.levels.${k}.range`),
    color: COLOR_MAP[k],
    description: t(lang, `guide.levels.${k}.description`),
    who: t(lang, `guide.levels.${k}.who`),
  }));

  return (
    <div className="guide-content">
      <p className="guide-intro">{t(lang, "guide_intro")}</p>

      {levels.map((level) => (
        <div className="guide-level" key={level.key}>
          <div className="guide-level-header">
            <div className="guide-level-swatch" style={{ background: level.color, color: level.color }} />
            <span className="guide-level-name" style={{ color: level.color }}>
              {level.name}
            </span>
            <span className="guide-level-range" data-num>{level.range}</span>
          </div>
          <div className="guide-level-body">
            {level.description}
            <div className="guide-level-who">{level.who}</div>
          </div>
        </div>
      ))}

      <div className="guide-explainer">
        <h4>{t(lang, "guide.why_title")}</h4>
        {t(lang, "guide.why_body")}
      </div>

      <div className="guide-explainer">
        <h4>{t(lang, "guide.about_title")}</h4>
        {t(lang, "guide.about_body")}
      </div>
    </div>
  );
}
