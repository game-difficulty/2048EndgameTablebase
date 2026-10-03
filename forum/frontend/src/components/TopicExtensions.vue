<script setup>
import { inject, ref, watch } from "vue";
import { api } from "../api";
const props = defineProps({ data: Object }),
  emit = defineEmits(["updated"]),
  { user, boards } = inject("forum");
const choices = ref([]),
  error = ref(""),
  busy = ref(false),
  reason = ref(""),
  questionStatus = ref("open"),
  answer = ref(""),
  duplicate = ref(""),
  board = ref("");
watch(
  () => props.data,
  (d) => {
    choices.value = d.poll?.choices || [];
    questionStatus.value = d.topic.question_status;
    answer.value = d.topic.accepted_post_id || "";
    duplicate.value = d.topic.duplicate_of || "";
    board.value = d.topic.board_id;
  },
  { immediate: true },
);
async function run(path, method, body) {
  busy.value = true;
  error.value = "";
  try {
    await api(path, { method, body });
    emit("updated");
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
  }
}
function question() {
  run(`/topics/${props.data.topic.id}/question`, "PUT", {
    status: questionStatus.value,
    accepted_post_id: Number(answer.value) || null,
    duplicate_of: Number(duplicate.value) || null,
    reason: reason.value,
    revision: props.data.topic.revision,
  });
}
</script>
<template>
  <p v-if="error" class="notice error" role="alert">{{ error }}</p>
  <section v-if="data.poll" class="panel">
    <h2>
      主题投票
      <span class="badge">{{ data.poll.ended ? "已结束" : "进行中" }}</span>
    </h2>
    <p class="muted">
      截止 {{ new Date(data.poll.closes_at).toLocaleString() }} · 最多选
      {{ data.poll.max_choices }} 项
    </p>
    <fieldset
      class="editor-fieldset"
      :disabled="busy || !user || data.poll.ended || data.topic.locked"
    >
      <label
        v-for="(option, i) in data.poll.options"
        :key="i"
        class="poll-option"
        ><input type="checkbox" v-model="choices" :value="i" />{{ option
        }}<span v-if="data.poll.counts">
          · {{ data.poll.counts[i] }} 票</span
        ></label
      ><button
        v-if="user"
        :disabled="!choices.length || choices.length > data.poll.max_choices"
        @click="run(`/topics/${data.topic.id}/vote`, 'PUT', { choices })"
      >
        提交 / 更新选择
      </button>
    </fieldset>
    <p class="muted">
      {{
        data.poll.counts
          ? `${data.poll.voters} 人参与`
          : "结果将在投票后或截止时显示。"
      }}
    </p>
    <button
      v-if="
        user &&
        !data.poll.ended &&
        (user.id === data.topic.author_id || data.topic.can_moderate)
      "
      :disabled="busy"
      @click="
        run(`/topics/${data.topic.id}/poll/close`, 'POST', {
          reason: '作者或管理人员提前结束投票',
        })
      "
    >
      提前结束投票
    </button>
  </section>
  <section v-if="data.topic.kind === 'question'" class="panel">
    <h2>
      问答与反馈
      <span class="badge">{{
        { open: "待解决", solved: "已解决", closed: "已关闭" }[
          data.topic.question_status
        ]
      }}</span>
    </h2>
    <RouterLink
      v-if="data.topic.accepted_post_id"
      :to="{ hash: '#p-' + data.topic.accepted_post_id }"
      >查看已采纳答案</RouterLink
    ><RouterLink
      v-if="data.topic.duplicate_of"
      :to="'/t/' + data.topic.duplicate_of"
    >
      · 查看关联问题</RouterLink
    >
    <details
      v-if="
        user && (user.id === data.topic.author_id || data.topic.can_moderate)
      "
    >
      <summary>更新问答状态</summary>
      <label class="field"
        >状态<select v-model="questionStatus">
          <option value="open">待解决</option>
          <option value="solved">已解决</option>
          <option value="closed">已关闭</option>
        </select></label
      ><label class="field"
        >采纳答案<select v-model="answer">
          <option value="">暂不指定答案</option>
          <option
            v-if="
              data.topic.accepted_post_id &&
              !data.posts.some((p) => p.id === data.topic.accepted_post_id)
            "
            :value="data.topic.accepted_post_id"
          >
            保留当前已采纳答案
          </option>
          <option
            v-for="post in data.posts.filter(
              (p) =>
                p.post_number > 1 && p.status === 'published' && !p.blocked,
            )"
            :key="post.id"
            :value="post.id"
          >
            #{{ post.post_number }} · {{ post.display_name }}
          </option></select
        ><small class="muted"
          >可选择已加载的回复，未出现时先加载对应楼层。</small
        ></label
      ><label class="field"
        >关联重复问题 ID<input
          v-model="duplicate"
          type="number"
          min="1" /></label
      ><label class="field"
        >修改说明<textarea v-model="reason" maxlength="1000" /></label
      ><button :disabled="busy || reason.trim().length < 3" @click="question">
        保存状态
      </button>
    </details>
  </section>
  <details v-if="data.topic.can_moderate" class="panel">
    <summary>移动到其他板块</summary>
    <label class="field"
      >目标板块<select v-model="board">
        <option v-for="b in boards" :key="b.id" :value="b.id">
          {{ b.name }}
        </option>
      </select></label
    ><label class="field"
      >移帖原因<textarea v-model="reason" maxlength="1000" /></label
    ><button
      :disabled="busy || reason.trim().length < 3"
      @click="
        run(`/topics/${data.topic.id}/move`, 'POST', {
          board_id: Number(board),
          revision: data.topic.revision,
          reason,
        })
      "
    >
      移帖
    </button>
  </details>
</template>
