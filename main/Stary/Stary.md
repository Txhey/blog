# Stary (v3.1.1)：个人网站项目

`github.io版`

## 功能介绍

* blog 笔记管理（存储、索引、查看）
  * view 查看笔记 (还可查看该笔记的其他附件 pdf等)
    * 左边是文件目录，中间主要查看的文件(markdown)内容，右上角悬浮菜单目录，右边是副浏览页，右上角可关闭。
    * 点击左边的文件，比如pdf后，将会在右边展示以作参考。
    * 点击md文件内容的其他笔记内容链接时，也将展示到右边。
      * 实现：每次查看链接是否是note仓库的文件，如果是，则将链接更改为当前连接并添加参数：referenceFilePath
* plan 计划管理  (每日计划、目标计划等)
* me 个人简历
* home 主页



## 页面设计

整体采用上写结构，主体固定最小宽度，不再采用百分比形式，减少了很多bug的产生。



## Vue项目实现

### 整体结构设计

#### 第三方库的使用

* 路由：`vue-router@4`
* 跨组件通信：`provide`和`inject`
* 样式语法标准：`sass`
* http访问：`axios`
* md文件预览：`md-editor-v3`
* 自定义svg图片：`vite-plugin-svg-icons`

#### 文件夹结构

每个组件都是一个文件夹，每个组件都包含一个子组件文件夹和一个模块组件

* components
	- component1
	  - subComponent (放子组件)
	    - subCoponent1
	      - subComponent (子组件也会有子组件)
	      - module (子组件也会有模块组件)
	      - `subComponent1.vue`
	  - module (放模块组件)
	  - `component1.vue`
	- component2 文件夹同上





### 下载、配置、修改内容

#### 使用`vite`创建一个全新vue3+js项目

```bash
# 创建vite项目
npm create vite@latest stary_v3.1.1

cd stary_v3.1.1
npm install
npm run dev
```



#### 下载依赖库

```bash
npm add vue-router@4
npm add axios
npm add sass
npm add element-plus
npm add vite-plugin-svg-icons
npm add md-editor-v3
npm i pdf-vue3
```



#### 配置`vite.config.js`

* 配置自定义svg图片的配置项 (图片路径，名字格式)
* 路径转化：将`src`文件夹映射成`@`，防止打包后的路径问题

```js
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

import { createSvgIconsPlugin } from 'vite-plugin-svg-icons'
import path from 'path'

export default defineConfig({
  plugins: [
    vue(),
    createSvgIconsPlugin({
      // 指定需要缓存的图标文件夹
      iconDirs: [path.resolve(process.cwd(), 'src/assets/svg')],
      // 指定symbolId格式
      symbolId: 'icon-[name]',
    })
  ],
  // 路径转化
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src/'),
    }
  },
})
```



#### 修改`index.html`文件

* 修改缩略图
  * 删除原有的``vite.svg`,添加自己的`Stary.svg`
* 修改标签页内容

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <!-- 修改缩略图标，图片放在public文件夹中 -->
    <link rel="shortcut icon" href="/Stary.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <!-- 修改标签页内容 -->
    <title>Stary</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.js"></script>
  </body>
</html>
```



#### 配置整体样式文件`base.css`

默认在`main.js`中导入使用的是`style.css`,这里我们修改成自己的`base.css`

1. 删除`style.css`文件
2. 创建`src/assets/css/base.css`文件

配置全局样式

```css
:root {
    --color-grey-0: #FAFAFA;
    --color-grey-1: #F5F5F5;
    --color-grey-2: #EEEEEE;
    --color-grey-3: #E0E0E0;
    --color-grey-4: #BDBDBD;
    --color-grey-5: #9E9E9E;
    --color-grey-6: #757575;
    --color-grey-7: #616161;
    --color-grey-8: #424242;
    --color-grey-9: #212121;

    --color-black-0: #808080;
    --color-black-1: #737373;
    --color-black-2: #666666;
    --color-black-3: #5A5A5A;
    --color-black-4: #4D4D4D;
    --color-black-5: #404040;
    --color-black-6: #333333;
    --color-black-7: #262626;
    --color-black-8: #1A1A1A;
    --color-black-9: #0D0D0D;
    --color-black-10: #000000;

    --color-red-0: #FFEBEE;
    --color-red-1: #FFCDD2;
    --color-red-2: #EF9A9A;
    --color-red-3: #E57373;
    --color-red-4: #EF5350;
    --color-red-5: #F44336;
    --color-red-6: #E53935;
    --color-red-7: #D32F2F;
    --color-red-8: #C62828;
    --color-red-9: #B71C1C;

    --color-blue-0: #E3F2FD;
    --color-blue-1: #BBDEFB;
    --color-blue-2: #90CAF9;
    --color-blue-3: #64B5F6;
    --color-blue-4: #42A5F5;
    --color-blue-5: #2196F3;
    --color-blue-6: #1E88E5;
    --color-blue-7: #1976D2;
    --color-blue-8: #1565C0;
    --color-blue-9: #0D47A1;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

span {
    color: var(--color-grey-0);
}
```



#### 配置路由``router`配置

创建文件：`src/router/index.js`

```js
import { createWebHashHistory, createRouter } from "vue-router";

import Index from '@/components/index/Index.vue'
import Home from "@/components/index/subComponents/home/Home.vue";
import Blog from "@/components/index/subComponents/blog/Blog.vue";
import Plan from "@/components/index/subComponents/plan/Plan.vue";
import Me from "@/components/index/subComponents/me/Me.vue";
import View from "@/components/index/subComponents/view/View.vue";

const routes = [
    {
        path: "/",
        redirect: "/index/home"
    },
    {
        path: "/index",
        component: Index,
        children: [
            {
                name: 'home',
                path: 'home',
                component: Home
            },
            {
                name: 'blog',
                path: 'blog',
                component: Blog
            },
            {
                name: 'plan',
                path: 'plan',
                component: Plan
            },
            {
                name: 'me',
                path: 'me',
                component: Me
            },
            {
                name: 'view',
                path: 'view',
                component: View,
            },
        ]
    }
]

const router = createRouter({
    history: createWebHashHistory(),
    routes,
});

export default router;
```



#### 配置自定义svg图片组件`SvgIcon.vue`

新建`src/assets/svg/SvgIcon.vue`文件

```vue
<template>
<svg @mouseover="hover" @mouseleave="leave" class="svgIcon" aria-hidden="true" :width="props.width"
     :height="props.height">
    <use :xlink:href="symbolId" :fill="tcolor" />
    </svg>
</template>

<script setup>
    import { ref, computed } from "vue";

    const props = defineProps({
        prefix: {
            type: String,
            default: "icon",
        },
        name: {
            type: String,
            required: true,
        },
        color: {
            type: String,
            default: "var(--color-text)",
        },
        hoverColor: {
            type: String,
            default: "var(--color-text-light)"
        },
        width: {
            type: String,
            default: "24px",
        },
        height: {
            type: String,
            default: "24px",
        },
    });

    const tcolor = ref(props.color);

    const symbolId = computed(() => `#${props.prefix}-${props.name}`);


    function hover() {
        tcolor.value = props.hoverColor;
    }

    function leave() {
        tcolor.value = props.color;
    }
</script>

<style lang="scss" scoped>
</style>
```

所有的svg图片都放到`src/assets/svg`文件夹中

##### 使用方式

```vue
<template>
<div>
    <SvgIcon name="Stary" width="4rem" height="4rem" color="#fff" hoverColor="#999"> </SvgIcon>
    </div>
</template>

<script setup>
    import SvgIcon from "@/assets/svg/SvgIcon.vue";
</script>
```



#### 配置`main.js`文件

```js
import { createApp } from 'vue'
import App from './App.vue'

// 样式
import '@/assets/css/base.css'
// router
import router from './router/index'
// svg 相关
import 'virtual:svg-icons-register'
// Element-plus
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";

const app = createApp(App);
app.use(router);
app.use(ElementPlus);
app.mount('#app')
```



#### 修改`App.vue`文件

内容全部删掉，只放一个`router-view`用作路由替换

```vue
<template>
  <div class="app">
    <router-view />
  </div>
</template>

<script setup>
</script>

<style lang="scss" scoped>
.app {
  background-color: var(--color-bg);
}
</style>
```



### 功能实现

#### 跨组件通信（`provide`，`inject`）

祖宗组件

```js
import { ref, provide } from 'vue'

const currentMenuName = ref('')
provide('currentMenuName', currentMenuName)
```

子组件

```js
import { ref, inject } from 'vue'

const currentMenuName = inject('currentMenuName')
currentMenuName.value = 'home'
```





#### 自定义主题

```css
:root{
    --color-base-blue: #469bf5;

    /* 背景色 */
    --color-bg: var(--color-black-7);

    /* 主题色 */
    --color-theme-light: var(--color-black-6);
    --color-theme-dark: var(--color-black-8);

    /* 字体色 */
    --color-text: var(--color-grey-5);
    --color-text-light: var(--color-grey-0);
    --color-text-dark: var(--color-grey-7);
    --color-span-selected: var(--color-blue-9);
    --color-span-bg: var(--color-grey-8);

    /* 边框颜色 */
    --color-border: var(--color-grey-5);
    --color-border-light: var(--color-grey-3);
    --color-border-dark: var(--color-grey-7);

    --color-scrollbar-trach: var(--color-grey-9);
    --color-scrollbar-thumb: var(--color-grey-8);

    --color-tag-bg: var(--color-grey-8);
    --color-tag-text: var(--color-blue-7);
    --color-tag-text-unselected: var(--color-grey-5);
}

body {
    background-color: var(--color-bg);
}
```





#### 路由跳转 (`router`)

```js
import { useRouter } from "vue-router";
const router = useRouter();

// 普通跳转
router.push("/index/home");

// 带参数并另外使用空白标签跳转
const { href } = router.resolve({
    name: 'view',
    query: {
        title: 'title',
        fileName: 'fileName.md'
    }
})
window.open(href, '_blank')
```





#### PDF预览 (`pdf-vue3`)

```vue
<template>
    <div class="pdf">
        <PDF src="https://mozilla.github.io/pdf.js/web/compressed.tracemonkey-pldi-09.pdf" />
    </div>
</template>

<script setup>
import PDF from "pdf-vue3";
</script>

<style lang="scss" scoped>
.pdf{
    width: 1000px;
}
</style>
```





#### 点击网址不跳转

```vue
<template>
    <div>
        <!-- 使用 md-editor-v3 渲染 Markdown 内容 -->
        <MdPreview v-model="markdownContent" previewOnly @onHtmlChanged="handleHtmlChange" />
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { MdPreview, MdCatalog } from "md-editor-v3";
import 'md-editor-v3/lib/style.css';

const markdownContent = ref(`# 示例 Markdown

[访问 Baidu](https://www.baidu.com)

[访问 Example](https://www.example.com)

[访问 Another Example](https://www.anotherexample.com)

[GitHub API Example](https://api.github.com/users/octocat)
`);

const handleHtmlChange = () => {
  const links = document.querySelectorAll('.md-editor-preview a');
  links.forEach(link => {
    // 移除之前添加的事件监听器
    link.removeEventListener('click', handleClick);
    // 添加新的事件监听器
    link.addEventListener('click', handleClick);
  });
};

const handleClick = (event) => {
  const url = event.target.href;

  if (url === 'https://www.baidu.com') {
    // 如果链接是 www.baidu.com，就正常跳转
    return;
  } else if (url.startsWith('https://api.github.com')) {
    // 如果链接以 https://api.github.com 开头，打印内容
    event.preventDefault();
    console.log(`链接地址是: ${url}`);
  } else {
    // 其他链接也阻止默认行为
    event.preventDefault();
    console.log(`其他链接地址: ${url}`);
  }
};

// 在组件挂载时处理初始渲染内容的链接
onMounted(handleHtmlChange);
</script>

<style>
a {
    color: blue;
    text-decoration: underline;
}
</style>
```







#### 样式笔记

```scss
span{
    // 能够使得无法选中文字内容
	user-select: none;
}
```

