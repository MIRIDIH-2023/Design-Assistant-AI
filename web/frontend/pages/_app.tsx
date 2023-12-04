import { Global, MantineProvider } from "@mantine/core";
import "../styles/globals.css";
import type { AppProps } from "next/app";
import { AppShell } from "@mantine/core";
import { Header } from "../components/Header";

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <>
      <MantineProvider withGlobalStyles withNormalizeCSS>
        <Global
          styles={[
            {
              "@font-face": {
                fontFamily: "Pretendard",
                src: `url('/fonts/Pretendard-Regular.woff2') format("woff2")`,
                fontWeight: 400,
                fontStyle: "normal",
              },
            },
            {
              "@font-face": {
                fontFamily: "Pretendard",
                src: `url('/fonts/Pretendard-Bold.woff2') format("woff2")`,
                fontWeight: 700,
                fontStyle: "normal",
              },
            },
            {
              "@font-face": {
                fontFamily: "Pretendard",
                src: `url('/fonts/Pretendard-Black.woff2') format("woff2")`,
                fontWeight: 900,
                fontStyle: "normal",
              },
            },
          ]}
        />
        <AppShell
          padding="md"
          header={
            <Header
              links={[
                { link: "/", label: "소개 " },
                { link: "/generator", label: "생성" },
              ]}
            />
          }
          styles={(theme) => ({
            main: {
              backgroundColor: theme.colors.gray[0],
            },
          })}
        >
          <Component {...pageProps} />
        </AppShell>
      </MantineProvider>
    </>
  );
}

export default MyApp;
