import axios from "axios";
import qs from "querystring";

async function detectLang(term: string) {
  if (term == null) {
    throw new Error("Search term should be provided as detectLang arguments");
  }

  const url = "papago/detectLangs";

  const params = qs.stringify({
    query: term,
  });

  const config = {
    baseURL: "https://openapi.naver.com/v1/",
    headers: {
      "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
      "x-naver-client-id": process.env.DETECT_CLIENT_ID,
      "x-naver-client-secret": process.env.DETECT_CLIENT_SECRET,
    },
  };

  const response = await axios.post(url, params, config);

  return response.data.langCode;
}

export default detectLang;
