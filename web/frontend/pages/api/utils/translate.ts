// pages/api/translate.js

import axios from "axios";
import { NextApiRequest, NextApiResponse } from "next";
import { NextRequest, NextResponse } from "next/server";
import qs from "querystring";

export async function POST(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "Only POST requests are accepted",
    });
  }

  const { term, original, transLang } = await req.body;

  if (!term) {
    return res.status(400).json({
      error: "Search term should be provided as translator arguments",
    });
  }

  const url = "papago/n2mt";

  const params = qs.stringify({
    source: original,
    target: transLang,
    text: term,
  });

  const config = {
    baseURL: "https://openapi.naver.com/v1/",
    headers: {
      "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
      "x-naver-client-id": process.env.TRANS_CLIENT_ID,
      "x-naver-client-secret": process.env.TRANS_CLIENT_SECRET,
    },
  };

  try {
    const response = await axios.post(url, params, config);
    return res.status(200).json({
      translatedText: response.data.message.result.translatedText,
    });
  } catch (error) {
    return res.status(500).json({
      error: error.message,
    });
  }
}

export default POST;
