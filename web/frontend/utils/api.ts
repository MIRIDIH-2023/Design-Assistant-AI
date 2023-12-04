import { parseLayout } from "./parser";

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
