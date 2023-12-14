import { parseLayout } from "./parser";

/**
 * 주어진 문장을 영어로 번역하고, 번역된 문장을 UDOP API에 전달하여 레이아웃을 예측합니다.
 * @param str_list 문장 리스트
 * @param img_url UDOP API에 전달할 이미지 URL
 * @returns 레이아웃 예측 결과
 */
export async function predictLayout(str_list: string[], img_url?: string) {
  const trans_list = [];
  for (let i = 0; i < str_list.length; i++) {
    trans_list.push(await translate(str_list[i], "ko", "en"));
  }
  const res: { layout: string } = await fetch(
    process.env.NEXT_PUBLIC_UDOP_API_ENDPOINT,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: trans_list,
        url: img_url,
      }),
    }
  ).then((res) => res.json());
  return parseLayout(res.layout);
}

/**
 * 주어진 문장을 SBERT API에 전달하여 이미지를 추천받습니다.
 * @param str_list 문장 리스트
 * @param num 추천받을 이미지의 개수
 * @returns 추천받은 이미지 URL 리스트
 */
export async function getBackgroundImages(str_list: string[], num: number) {
  const res: { image_url: string[] } = await fetch(
    process.env.NEXT_PUBLIC_SBERT_API_ENDPOINT,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: str_list,
        num_recommend: num,
        use_dalle: false,
      }),
    }
  ).then((res) => res.json());
  return res.image_url;
}

/**
 * 주어진 문장을 파파고 API를 이용하여 번역합니다.
 * @param term 번역할 문장
 * @param original 번역할 문장의 언어
 * @param transLang 번역될 언어
 * @returns 번역된 문장
 */
export async function translate(
  term: string,
  original: string,
  transLang: string
) {
  const res: { translatedText: string } = await fetch(
    "http://localhost:3000/api/utils/translate",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        term: term,
        original: original,
        transLang: transLang,
      }),
    }
  ).then((res) => res.json());

  return res.translatedText;
}
