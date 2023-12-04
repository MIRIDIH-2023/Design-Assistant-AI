namespace NodeJS {
  interface ProcessEnv extends NodeJS.ProcessEnv {
    TRANS_CLIENT_ID: string;
    TRANS_CLIENT_SECRET: string;
    DETECT_CLIENT_ID: string;
    DETECT_CLIENT_SECRET: string;
    OPENAI_API_KEY: string;
    NEXT_PUBLIC_UDOP_API_ENDPOINT: string;
    NEXT_PUBLIC_SBERT_API_ENDPOINT: string;
  }
}
